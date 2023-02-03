import numpy as np
import numpy as np
import os
import sys

from scipy.spatial.transform import Rotation as R

if '/opt/drake/lib/python3.8/site-packages' not in sys.path:
    sys.path.append('/opt/drake/lib/python3.8/site-packages')


from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser
)
from pydrake.all import (
    DiagramBuilder,  Parser,  RigidTransform, RotationMatrix, Solve
)

from pydrake.multibody import inverse_kinematics
from pydrake.all import Parser

from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, 
    DiagramBuilder, CoulombFriction,
    Parser,  RigidTransform,
    RotationMatrix, Solve, Box, SpatialInertia, UnitInertia
)
from pydrake.multibody import inverse_kinematics
from pydrake.common import RandomGenerator

from pydrake.planning.common_robotics_utilities import (
    MakeKinematicLinearRRTNearestNeighborsFunction,
    MakeKinematicLinearBiRRTNearestNeighborsFunction,
    MakeRRTTimeoutTerminationFunction,
    MakeBiRRTTimeoutTerminationFunction,
    PropagatedState,
    RRTPlanSinglePath,
    BiRRTPlanSinglePath,
    SimpleRRTPlannerState,
)
import time
import random

def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def AddMovo(plant, movable_joint_names, q0, scene_graph=None, verbose=True):
    """ rewrote AddIiwa() in https://github.com/RussTedrake/manipulation/blob/master/manipulation/scenarios.py
        combine with AddJointActuator() examples in AddPlanarGripper() in manipulation/forces.ipynb
    """

    mu = 0.5
    mass = 1.0
    shape = Box(0.6, 1.2, 0.03)
    X_WBox = RigidTransform([1, 0, 0.73])
    inertia = UnitInertia.SolidBox(shape.width(), shape.depth(), shape.height())

    name = "table"
    instance = plant.AddModelInstance(name)
    body = plant.AddRigidBody(name, instance, SpatialInertia(mass=mass, p_PScm_E=np.array([0.,0.,0.]), G_SP_E=inertia))
    plant.RegisterCollisionGeometry(body, X_WBox, shape, name, CoulombFriction(mu, mu))
    plant.RegisterVisualGeometry(body, X_WBox, shape, name, [.9, .9, .9, 1.0])

    model_file = FindResource("../models/srl/movo_description_drake/movo_robotiq_collision.urdf")
    parser = Parser(plant)
    parser.package_map().Add("movo_description_drake", "./models/srl/movo_description_drake")
    movo_model = parser.AddModelFromFile(model_file)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("world_link"))

    print("Plant geometry registered")
    print(plant.geometry_source_is_registered())
    """ without the above line there will be error:
    
            RuntimeError: DiagramBuilder::Connect: Mismatched vector sizes while connecting 
            output port iiwa7_continuous_state of System plant (size 27) to input port u0 of 
            System drake/systems/Demultiplexer@000055f8a5390400 (size 28)
            
        with a different link name there will be error:
            
            RuntimeError: This mobilizer is creating a closed loop since the outboard body 
            already has an inboard mobilizer connected to it. If a physical loop is really needed, 
            consider using a constraint instead.
    """
    index = 0
    for joint_name in movable_joint_names:
        print(joint_name)
        joint = plant.GetJointByName(joint_name)
        if joint.type_name() == 'prismatic':  ## gripper
#             joint.set_default_translation(0)
            joint.set_default_translation(q0[index])
            index += 1
        elif joint.type_name() == 'weld':
            continue
        elif joint.type_name() in ['revolute', 'continuous']:  ## arm
#             joint.set_default_angle(0)
            joint.set_default_angle(q0[index])
            index += 1
        else:
            raise NotImplementedError
            
        plant.AddJointActuator(joint_name, joint)


    plant.Finalize()
    return movo_model

def equal(current_q, target_q, epsilon = 0.05):
    if isinstance(target_q, tuple):
        target_q = np.asarray(target_q)

    return np.linalg.norm(current_q - target_q) < epsilon


def drake_ik(xyz, quat, movable_joint_names, q0, verbose=True, tool_link="right_tool_link"):
    """ Convert end-effector pose list to joint position list using series of
    InverseKinematics problems. Note that q is 9-dimensional because the last 2 dimensions
    contain gripper joints, but these should not matter to the constraints.
    @param: pose_lst (python list): post_lst[i] contains keyframe X_WG at index i.
    @return: q_knots (python_list): q_knots[i] contains IK solution that will give f(q_knots[i]) \approx pose_lst[i].
    """

    if isinstance(quat, RotationMatrix):
        r = quat
    else:
        r = R.from_quat(quat)
        r = RotationMatrix(r.as_matrix())

    desired_pose = RigidTransform(r, xyz)

    q_knots = []
    
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    panda = AddMovo(plant, movable_joint_names, q0)

    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName(tool_link)
    q_nominal = q0
    # if verbose: print('len(q_nominal)', len(q_nominal))
    def AddOrientationConstraint(ik, R_WG, bounds):
        """Add orientation constraint to the ik problem. Implements an inequality
        constraint where the axis-angle difference between f_R(q) and R_WG must be
        within bounds. Can be translated to:
        ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
        """
        ik.AddOrientationConstraint(
                frameAbar=world_frame, R_AbarA=R_WG,
                frameBbar=gripper_frame, R_BbarB=RotationMatrix(),
                theta_bound=bounds
        )

    def AddPositionConstraint(ik, p_WG_lower, p_WG_upper):
        """Add position constraint to the ik problem. Implements an inequality
        constraint where f_p(q) must lie between p_WG_lower and p_WG_upper. Can be
        translated to
        ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
        """
        ik.AddPositionConstraint(
                frameA=world_frame, frameB=gripper_frame, p_BQ=np.zeros(3),
                p_AQ_lower=p_WG_lower, p_AQ_upper=p_WG_upper)

    ik = inverse_kinematics.InverseKinematics(plant)

    q_variables = ik.q() # Get variables for MathematicalProgram
    prog = ik.prog() # Get MathematicalProgram

    #### Modify here ###############################

    X_WG = desired_pose
    R_WG = X_WG.rotation()
    p_WG = X_WG.translation()

    ## an equality constraint for the constrained degrees of freedom
    AddPositionConstraint(ik, p_WG, p_WG)
    
    ## inequality constraints for the unconstrained one
    AddOrientationConstraint(ik, R_WG, 0.01)
    
    ## Add a joint-centering cost on q_nominal
    prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_nominal, q_variables)

    ## Set initial guess to be nominal configuration

    prog.SetInitialGuess(q_variables, q_nominal)

    ################################################

    result = Solve(prog)

    if not result.is_success():
        return None
        
    q = list(result.GetSolution(q_variables))

    return q


from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
import math


def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return (
        np.less_equal(lower_limits, values).all()
        and np.less_equal(values, upper_limits).all()
    )

def check_collision(plant, plant_context, query_object, positions):
    new_positions = plant.GetPositions(plant_context)
    new_positions[:len(positions)] = positions
    plant.SetPositions(plant_context, np.array(new_positions))
    return query_object.HasCollisions()


def test_birrt(plant, plant_context, query_object, qs, qg, lower, upper):
    seed = 0
    start = np.array(qs)
    goal = np.array(qg)
    goal_bias = 0.2
    step_size = math.radians(10)
    check_step = math.radians(0.1)
    solve_timeout = 60

    start_tree = [SimpleRRTPlannerState(start)]
    goal_tree = [SimpleRRTPlannerState(goal)]

    def sampling_fn():
        if (np.random.rand() < goal_bias):
            return goal
        return [random.uniform(l, u) for (l, u) in zip(lower, upper)]

    def distance_fn(point1, point2):
        return np.linalg.norm(point2 - point1)

    def extend_fn(nearest, sample, is_start_tree):
        if(not all_between(lower, sample, upper)):
            return []

        extend = None
        extend_dist = distance_fn(nearest, sample)
        if extend_dist <= step_size:
            extend = sample
        else:
            extend = nearest + step_size/extend_dist * (sample - nearest)

        check_dist = distance_fn(nearest, extend)
        for ii in range(1, int(check_dist/check_step)):
            check_point = nearest  + ii * check_step / check_dist * (extend - nearest)
            if(check_collision(plant, plant_context, query_object, check_point)):
                return []
        return [PropagatedState(state=extend, relative_parent_index=-1)]

    def connect_fn(nearest, sample, is_start_tree):
         
        if(not all_between(lower, sample, upper)):
            return []
        total_dist = distance_fn(nearest, sample)
        total_steps = int(np.ceil(total_dist / step_size))

        propagated_states = []
        parent_offset = -1
        current = nearest
        for steps in range(total_steps):
            current_target = None
            target_dist = distance_fn(current, sample)
            if (target_dist > step_size):
                current_target = current \
                    + step_size/target_dist * (sample - current)
            elif (target_dist < 1e-6):
                break
            else:
                current_target = sample

            check_dist = distance_fn(current, current_target)
            for ii in range(1, int(check_dist/check_step)):
                check_point = current + ii * check_step / check_dist \
                    * (current_target - current)
                if(check_collision(plant, plant_context, query_object, check_point)):
                    return propagated_states
            propagated_states.append(PropagatedState(
                state=current_target, relative_parent_index=parent_offset))
            parent_offset += 1
            current = current_target

        return propagated_states

    def states_connected_fn(source, target, is_start_tree):
        return np.linalg.norm(source - target) < 1e-6

    nearest_neighbor_fn = MakeKinematicLinearBiRRTNearestNeighborsFunction(
        distance_fn=distance_fn, use_parallel=False)

    termination_fn = MakeBiRRTTimeoutTerminationFunction(solve_timeout)

    extend_result = BiRRTPlanSinglePath(
        start_tree=start_tree, goal_tree=goal_tree,
        state_sampling_fn=sampling_fn,
        nearest_neighbor_fn=nearest_neighbor_fn, propagation_fn=extend_fn,
        state_added_callback_fn=None,
        states_connected_fn=states_connected_fn,
        goal_bridge_callback_fn=None,
        tree_sampling_bias=0.5, p_switch_tree=0.25,
        termination_check_fn=termination_fn, rng=RandomGenerator(seed))

    print(extend_result.Path())

    connect_result = BiRRTPlanSinglePath(
        start_tree=start_tree, goal_tree=goal_tree,
        state_sampling_fn=sampling_fn,
        nearest_neighbor_fn=nearest_neighbor_fn, propagation_fn=connect_fn,
        state_added_callback_fn=None,
        states_connected_fn=states_connected_fn,
        goal_bridge_callback_fn=None,
        tree_sampling_bias=0.5, p_switch_tree=0.25,
        termination_check_fn=termination_fn, rng=RandomGenerator(seed))

    return connect_result.Path()

def test_rrt(plant, plant_context, query_object, qs, qg, lower, upper):
    start = np.array(qs)
    goal = np.array(qg)
    goal_bias = 0.2
    step_size = math.radians(10)
    check_step = math.radians(0.1)
    solve_timeout = 60

    rrt_tree = [SimpleRRTPlannerState(start)]
    def sampling_fn():
        if (np.random.rand() < goal_bias):
            return goal
        return [random.uniform(l, u) for (l, u) in zip(lower, upper)]

    def distance_fn(point1, point2):
        return np.linalg.norm(point2 - point1)

    def check_goal_fn(sample):
        return np.linalg.norm(sample - goal) < 1e-6

    def extend_fn(nearest, sample):

        if(not all_between(lower, sample, upper)):
            return []

        extend = None
   
        extend_dist = distance_fn(nearest, sample)
      
        if extend_dist <= step_size:
            extend = sample
        else:
            extend = nearest + step_size/extend_dist * (sample - nearest)

        check_dist = distance_fn(nearest, extend)
        for ii in range(1, int(check_dist/check_step)):
            check_point = nearest \
                + ii * check_step / check_dist * (extend - nearest)
            if(check_collision(plant, plant_context, query_object, check_point)):
                return []
        return [PropagatedState(state=extend, relative_parent_index=-1)]

    nearest_neighbor_fn = MakeKinematicLinearRRTNearestNeighborsFunction(
        distance_fn=distance_fn, use_parallel=False)

    termination_fn = MakeRRTTimeoutTerminationFunction(solve_timeout)

    single_result = RRTPlanSinglePath(
        tree=rrt_tree, sampling_fn=sampling_fn,
        nearest_neighbor_fn=nearest_neighbor_fn,
        forward_propagation_fn=extend_fn, state_added_callback_fn=None,
        check_goal_reached_fn=check_goal_fn, goal_reached_callback_fn=None,
        termination_check_fn=termination_fn)
    print(single_result)

    return single_result.Path()

def drake_motion_planning(qs, qg, movable_joint_names, lower, upper, obstacles=[]):
    # sim_timestep = 1e-3
    # plant = MultibodyPlant(sim_timestep)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    panda = AddMovo(plant, movable_joint_names, qs)

    proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
    visualizer = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url, delete_prefix_on_load=False)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant_qs = plant.GetPositions(plant_context)

    query_object = plant.get_geometry_query_input_port().Eval(plant_context)
    inspector = query_object.inspector()

    visualizer.load()
    diagram.Publish(context)


    new_positions = plant.GetPositions(plant_context)
    new_positions[:len(qg)] = qg
    plant.SetPositions(plant_context, new_positions)

    print("Collisions: "+str(query_object.HasCollisions()))

    if(query_object.HasCollisions()):
        return None

    start_time = time.time()

    rrt_result = test_birrt(plant, plant_context, query_object, qs, qg, lower, upper)
    print("Planning time: "+str(time.time()-start_time))

    # time.sleep(10)

    for positions in rrt_result:
        collision = check_collision(plant, plant_context, query_object, positions)
        # diagram.Publish(context)
        print(collision)      

    return rrt_result


if(__name__=="__main__"):
    xyz = [0.5599999998462588, 0.12000000138395996, 0.3799999947283723]
    quat = [ -0.4608969, 0.0483233, 0.7118954, 0.5276779 ]
    print(drake_ik(xyz, quat))