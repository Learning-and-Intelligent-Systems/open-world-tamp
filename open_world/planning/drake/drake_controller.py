import numpy as np
import numpy as np
import os
import sys

import importlib
from scipy.spatial.transform import Rotation as R

import copy

if '/opt/drake/lib/python3.8/site-packages' not in sys.path:
    sys.path.append('/opt/drake/lib/python3.8/site-packages')


import pydrake
from pydrake.all import (
    RotationMatrix, Solve, AddMultibodyPlantSceneGraph,
    MultibodyPlant, Parser, RigidTransform
)

from pydrake.multibody import inverse_kinematics
from pydrake.all import Parser
from pydrake.multibody import inverse_kinematics
from pydrake.all import MultibodyPlant

def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def CreateMovoControllerPlant(movable_joint_names, q0, **kwargs):
    sys.path.append('/opt/drake/lib/python3.8/site-packages')
    sim_timestep = 1e-3
    plant_robot = MultibodyPlant(sim_timestep)
    AddMovo(plant_robot, movable_joint_names, q0, **kwargs)
    plant_robot.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    plant_robot.Finalize()
    return plant_robot

def AddMovo(plant, movable_joint_names, q0, scene_graph=None, verbose=True):
    """ rewrote AddIiwa() in https://github.com/RussTedrake/manipulation/blob/master/manipulation/scenarios.py
        combine with AddJointActuator() examples in AddPlanarGripper() in manipulation/forces.ipynb
    """

    model_file = FindResource("../models/srl/movo_description_drake/movo_robotiq_collision.urdf")
    panda_model = Parser(plant).AddModelFromFile(model_file)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("world_link"))

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
            
        plant.AddJointActuator(joint_name, joint)

   
    return panda_model

def equal(current_q, target_q, epsilon = 0.05):
    if isinstance(target_q, tuple):
        target_q = np.asarray(target_q)

    return np.linalg.norm(current_q - target_q) < epsilon


def CreateMultibodyPlantWithPandaController(dimensions, center, q0=None):

    manipulation = importlib.import_module("drake_examples.manipulation")
    from manipulation.scenarios import AddShape

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    panda = AddPanda(plant, scene_graph, q0=q0)
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])

    w, l, h = dimensions
    x, y, z = center
    box = AddShape(plant, Box(w, l, h), "box")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("box", box), RigidTransform([x, y, z]))
    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    q0 = plant.GetPositions(plant_context)
    print('len(q0)', len(q0))
    return plant, plant_context

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
    plant = CreateMovoControllerPlant(movable_joint_names, q0=q0, verbose=False)

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

if(__name__=="__main__"):
    xyz = [0.5599999998462588, 0.12000000138395996, 0.3799999947283723]
    quat = [ -0.4608969, 0.0483233, 0.7118954, 0.5276779 ]
    print(drake_ik(xyz, quat))