#!/usr/bin/env python

from __future__ import print_function

try:
    import pybullet as p
except ImportError:
    raise ImportError(
        "This example requires PyBullet (https://pypi.org/project/pybullet/)"
    )

import argparse
import cProfile
import os
import pstats
import sys

import numpy as np

from .constant import POSE_DIR, SEG_DIR, YCB_BANK_DIR

sys.path.extend(
    [
        "pddlstream",
        #'pybullet-planning',
        #'pddlstream/examples/pybullet/utils',
        POSE_DIR,
    ]
)

from examples.discrete_belief.run import MAX_COST, clip_cost, revisit_mdp_cost
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    Attach,
    Conf,
    Detach,
    Pose,
    Trajectory,
    apply_commands,
    get_base_limits,
    get_grasp_gen,
    get_ik_fn,
    get_ik_ir_gen,
    get_motion_gen,
    get_stable_gen,
)
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    ARM_NAMES,
    attach_viewcone,
    get_arm_joints,
    get_group_conf,
    get_group_joints,
    get_link_pose,
    is_drake_pr2,
)
from examples.pybullet.utils.pybullet_tools.transformations import (
    quaternion_from_matrix,
    quaternion_matrix,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    AABB,
    CLIENT,
    INFO_FROM_BODY,
    STATIC_MASS,
    ClientSaver,
    HideOutput,
    LockRenderer,
    ModelInfo,
    WorldSaver,
    add_body_name,
    add_data_path,
    clone_body,
    connect,
    create_box,
    create_cylinder,
    disable_gravity,
    disconnect,
    draw_aabb,
    draw_base_limits,
    enable_gravity,
    get_bodies,
    get_configuration,
    get_distance,
    get_joint_position,
    get_joint_positions,
    get_mesh_geometry,
    get_pose,
    has_gui,
    is_center_stable,
    load_pybullet,
    pairwise_collision,
    remove_body,
    save_image,
    set_client,
    set_configuration,
    set_euler,
    set_joint_positions,
    set_point,
    set_pose,
    step_simulation,
    unit_pose,
    wait_for_user,
)
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.search import ABSTRIPSLayer
from pddlstream.language.constants import And, Equal, Or, PDDLProblem, print_solution
from pddlstream.language.generator import (
    accelerate_list_gen_fn,
    fn_from_constant,
    from_fn,
    from_gen_fn,
    from_list_fn,
    from_test,
)
from pddlstream.language.stream import StreamInfo
from pddlstream.utils import get_file_path, read

from .primitives import (
    REG_RANGE,
    VIS_RANGE,
    Detect,
    Mark,
    Observe,
    Observe_specific,
    Register,
    Scan,
    ScanRoom,
    get_cone_commands,
    get_fo_test,
    get_in_range_test,
    get_inverse_visibility_fixbase_fn,
    get_inverse_visibility_fn,
    get_isGraspable_test,
    get_isPoseCertain_test,
    get_isTarget_test,
    get_unblock_test,
    get_vis_base_gen,
    get_visclear_test,
    move_look_trajectory,
    plan_head_traj,
)
from .problems import USE_DRAKE_PR2, create_pr2, get_problem1

BASE_CONSTANT = 1
BASE_VELOCITY = 0.5


"""  ================ modified function ================  """


def create_body(
    collision_id=-1,
    visual_id=-1,
    mass=STATIC_MASS,
    client=CLIENT,
    base_point=(0, 0, 0),
    base_quat=(0, 0, 0, 1.0),
):
    # add base pose
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=base_point,
        baseOrientation=base_quat,
        physicsClientId=client,
    )


def create_shape(
    geometry, pose=unit_pose(), color=(1, 0, 0, 1), specular=None, client=CLIENT
):
    # add param: client
    point, quat = pose
    collision_args = {
        "collisionFramePosition": point,
        "collisionFrameOrientation": quat,
        "physicsClientId": client,
    }
    collision_args.update(geometry)
    if "length" in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args["height"] = collision_args["length"]
        del collision_args["length"]
    collision_id = p.createCollisionShape(**collision_args)

    if color is None:  # or not has_gui():
        return collision_id, NULL_ID
    visual_args = {
        "rgbaColor": color,
        "visualFramePosition": point,
        "visualFrameOrientation": quat,
        "physicsClientId": client,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args["specularColor"] = specular
    visual_id = p.createVisualShape(**visual_args)
    return collision_id, visual_id


def create_obj(
    path,
    scale=1.0,
    mass=STATIC_MASS,
    color=(0.5, 0.5, 0.5, 1),
    client=CLIENT,
    base_pose=unit_pose(),
):
    # add param client & pose
    collision_id, visual_id = create_shape(
        get_mesh_geometry(path, scale=scale), color=color, client=client
    )
    base_point, base_quat = base_pose
    body = create_body(
        collision_id,
        visual_id,
        mass=mass,
        client=client,
        base_point=base_point,
        base_quat=base_quat,
    )
    fixed_base = mass == STATIC_MASS
    INFO_FROM_BODY[client, body] = ModelInfo(
        None, path, fixed_base, scale
    )  # TODO: store geometry info instead?
    return body


def load_pybullet(filename, fixed_base=False, scale=1.0, client=CLIENT, **kwargs):
    # add param: client
    with LockRenderer():
        if filename.endswith(".urdf"):
            flags = get_urdf_flags(**kwargs)
            body = p.loadURDF(
                filename,
                useFixedBase=fixed_base,
                flags=flags,
                globalScaling=scale,
                physicsClientId=client,
            )
        elif filename.endswith(".sdf"):
            body = p.loadSDF(filename, physicsClientId=client)
        elif filename.endswith(".xml"):
            body = p.loadMJCF(filename, physicsClientId=client)
        elif filename.endswith(".bullet"):
            body = p.loadBullet(filename, physicsClientId=client)
        elif filename.endswith(".obj"):
            # TODO: fixed_base => mass = 0?
            body = create_obj(filename, scale=scale, client=client, **kwargs)
        else:
            raise ValueError(filename)
    INFO_FROM_BODY[client, body] = ModelInfo(None, filename, fixed_base, scale)
    return body


def clone_world(client=None, exclude=[], loadmesh=[]):
    # There is a bug(?) with pybullet getCollisionShapeData(can't retrieve
    # filenamefilename), so a list of [loadmesh] need to be provided for those
    # models created with p.createMultiBody
    visual = has_gui(client)
    mapping = {}
    for body in get_bodies():
        if body > 20:  # TODO adhoc
            continue
        if body not in exclude and body not in loadmesh:
            # print(f'Cloning obj {body}')
            new_body = clone_body(body, collision=True, visual=visual, client=client)
            mapping[body] = new_body
        elif (
            body in loadmesh
        ):  # modified by xiaolinf June 12 2020. pybullet cant get the value of '.obj' file through p.getCollisionShapeData
            # print(f'Cloning obj {body}')
            new_body = load_pybullet(
                loadmesh[body], client=client, base_pose=get_pose(body)
            )
            # newbody_vis = p.createVisualShape(p.GEOM_MESH,
            #                             fileName=loadmesh[body])
            # newbody_col = p.createCollisionShape(p.GEOM_MESH,
            #                             fileName=loadmesh[body])
            # new_body = p.createMultiBody(baseMass=1,baseCollisionShapeIndex=newbody_col,
            #                         baseVisualShapeIndex=newbody_vis,
            #                         physicsClientId=client,
            #                         basePosition=get_point(body),
            #                         baseOrientation=get_quat(body))
            mapping[body] = new_body
    return mapping


def get_sample_vs_gen(problem):  # sample target location to observe
    robot = problem.robot
    vspace = problem.vspace

    def fn(placeholder_vs):
        while True:
            # TODO clustering
            unknown_list = list(vspace.occupied)
            if len(unknown_list) == 0:
                return
                # break #return None
            idx = np.random.randint(len(unknown_list))
            target = unknown_list[idx]
            if target in vspace.poses.keys():
                yield (
                    vspace.poses[target],
                )  # avoid assigning two Pose object to one location
                break  # continue
            loc = vspace.center_from_voxel(target)
            pose = (tuple(loc), unit_quat())
            pose_obj = Pose(body=None, value=pose)
            vspace.poses[target] = pose_obj
            yield (pose_obj,)

    return fn


def get_ik_ir_fixbase(problem, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    # ir_sampler = get_ir_sampler(problem, learned=True, max_attempts=1,
    # **kwargs) # plan base motion
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)

    def gen(*inputs):
        b, a, p, g, bconf = inputs
        if isinstance(a, Entity):
            a = a.obj_id
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            ik_outputs = ik_fn(*(inputs))
            if ik_outputs is None:
                continue
            print("IK attempts:", attempts)
            yield ik_outputs
            return
            # if not p.init:
            #    return

    return gen


"""  =========================================================  """


def move_cost_fn(c):
    [t] = c.commands
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    return BASE_CONSTANT + distance / BASE_VELOCITY


def pddlstream_from_state(state, teleport=False):
    task = state.task

    # wait_for_user()

    robot = task.robot
    # TODO: infer open world from task

    domain_pddl = read(get_file_path(__file__, "domain.pddl"))
    stream_pddl = read(get_file_path(__file__, "stream.pddl"))
    constant_map = {
        "base": "base",
        "left": "left",
        "right": "right",
        "head": "head",
    }
    # constant_map = {}
    # base_conf = state.poses[robot]
    base_conf = Conf(
        robot, get_group_joints(robot, "base"), get_group_conf(robot, "base")
    )
    scan_cost = 3
    init = [
        (("CanMove",)),
        ("BConf", base_conf),
        ("AtBConf", base_conf),
        Equal(("MoveCost",), 1),
        Equal(("PickCost",), 1),
        Equal(("PlaceCost",), 1),
        Equal(("ScanCost",), scan_cost),
        Equal(("RegisterCost",), 1),
        ("Visible", task.vspace),  # TODO  observation test
    ]
    holding_arms = set()
    holding_bodies = set()
    for attach in state.attachments.values():
        holding_arms.add(attach.arm)
        holding_bodies.add(attach.child)
        init += [
            ("Grasp", attach.child, state.grasps[attach.child]),
            ("AtGrasp", attach.arm, attach.child, state.grasps[attach.child]),
        ]
        # holding_arms.add(attach.arm)
        # holding_bodies.add(attach.body)
        # init += [('Grasp', attach.body, attach.grasp),
        #          ('AtGrasp', attach.arm, attach.body, attach.grasp)]
    for arm in ARM_NAMES:
        joints = get_arm_joints(robot, arm)
        conf = Conf(robot, joints, get_joint_positions(robot, joints))
        init += [("Arm", arm), ("AConf", arm, conf), ("AtAConf", arm, conf)]
        # if arm in task.arms:
        init += [("Controllable", arm)]
        if arm not in holding_arms:
            init += [("HandEmpty", arm)]
    for body in task.get_bodies():  # TODO assume some known,fixed bodies
        if body in holding_bodies:
            continue
        # TODO: no notion whether observable actually corresponds to the correct thing
        pose = state.poses[body]
        init += [("Pose", body, pose), ("AtPose", body, pose)]

    # init += [('Scannable', body) for body in task.rooms + task.surfaces]
    # init += [('Registerable', body) for body in task.movable]
    init += [("Graspable", body) for body in task.movable]
    for body in task.get_bodies():
        supports = task.get_supports(body)
        if supports is None:
            continue
        for surface in supports:
            p_obs = state.b_on[body].prob(surface)
            cost = revisit_mdp_cost(
                0, scan_cost, p_obs
            )  # TODO: imperfect observation model
            init += [
                ("Stackable", body, surface),
                Equal(("LocalizeCost", surface, body), clip_cost(cost)),
            ]
            # if is_placement(body, surface):
            if is_center_stable(body, surface):
                if body in holding_bodies:
                    continue
                pose = state.poses[body]
                init += [("Supported", body, pose, surface)]
    for entity in state.registered.values():
        init.append(("Registered", entity))
        init += [("Pose", entity, entity.loc), ("AtPose", entity, entity.loc)]
        init += [("Stackable", entity, surface) for surface in task.get_bodies()]
        init += [("Graspable", entity)]  # TODO repetitive?
        if not entity.pose_uncertain:
            init += [("PoseCertain", entity)]
    # for body in task.get_bodies():
    #     if state.is_localized(body):
    #         init.append(('Localized', body))
    #     else:
    #         init.append(('Uncertain', body))
    #     if body in state.registered:
    #         init.append(('Registered', body))
    for block_obj in state.block_list.keys():  # observation test
        for (obj_loc, blocked_loc) in state.block_list[block_obj]:
            init_cmd = ("Block", block_obj, obj_loc, blocked_loc)
            init.append(init_cmd)
            init_cmd = ("Pose", block_obj, obj_loc)
            if init_cmd not in init:
                init.append(init_cmd)
    #     init_cmd = ('Registered',block_obj)
    #     if init_cmd not in init:
    #         init.append(init_cmd)
    # for vs in task.goal_fo:
    #     if vs.fully_observed():
    #         init.append(('Fo', vs))
    #     else:
    #         init.append(('Po',vs))
    if task.vspace.fully_observed():
        init.append(("Fo", task.vspace))
    else:
        init.append(("Po", task.vspace))

    has_target = False
    for entity in state.target:
        init += [("Targeted", entity), ("Target", entity)]
        has_target = True

    goal = (
        "Fo",
        task.vspace,
    )  # Or(*[('HoldTarget',)] + [('Fo', task.vspace)]) #And(*[('Holding', a, b) for a, b in task.goal_holding] + \

    # if has_target:
    #     goal = ('HoldTarget',)
    # elif len(state.registered.keys())>0:
    #     goal = Or(*[('Targeted', entity) for entity in
    #         state.registered.values()] + \
    #             [('Fo', task.vspace)])
    # elif task.vspace.fully_observed():
    #     goal = ('HoldTarget',)
    # else:
    #     goal = Or(*[('HoldTarget',)] + [('Fo', task.vspace)]) #And(*[('Holding', a, b) for a, b in task.goal_holding] + \
    # [('On', b, s) for b, s in task.goal_on] + \
    # [('Localized', b) for b in task.goal_localized] + \
    # [('Registered', b) for b in task.goal_registered] + \
    # [('Fo', grid) for grid in task.goal_fo] # observation test
    # )
    # for vspace in task.goal_fo:
    #     for key,box in vspace.box.items():
    #         goal += tuple([('Fo',box)])
    #         init += [('Visible',box)]

    stream_map = {
        "sample-pose": from_gen_fn(get_stable_gen(task)),
        "sample-grasp": from_list_fn(get_grasp_gen(task)),
        "inverse-kinematics-fixbase": from_gen_fn(
            get_ik_ir_fixbase(task, teleport=teleport)
        ),
        # 'inverse-kinematics': from_gen_fn(get_ik_ir_gen(task, teleport=teleport)),
        # 'plan-base-motion': from_fn(get_motion_gen(task, teleport=teleport)),
        # 'MoveCost': move_cost_fn,
        "TrajPoseCollision": fn_from_constant(False),
        "TrajArmCollision": fn_from_constant(False),
        "TrajGraspCollision": fn_from_constant(False),
        # 'test-vis-base': from_test(get_in_range_test(task, VIS_RANGE)),
        # 'test-reg-base': from_test(get_in_range_test(task, REG_RANGE)),
        # 'test-fully-observed': from_test(get_fo_test(task)),
        # 'sample-vis-base': accelerate_list_gen_fn(from_gen_fn(get_vis_base_gen(task, VIS_RANGE)), max_attempts=25),
        # 'sample-reg-base': accelerate_list_gen_fn(from_gen_fn(get_vis_base_gen(task, REG_RANGE)), max_attempts=25),
        # 'inverse-visibility': from_fn(get_inverse_visibility_fn(task)),
        "sample-vis": from_gen_fn(get_sample_vs_gen(task)),
        "inverse-visibility-fixbase": from_fn(get_inverse_visibility_fixbase_fn(task)),
        # 'make-obs': from_fn(make_obs(task)),
        "test-unblock": from_test(get_unblock_test(task)),
        "test-visclear": from_test(get_visclear_test(task)),
        "test-graspable": from_test(get_isGraspable_test(task)),
        "test-isTarget": from_test(get_isTarget_test(task)),
        "test-isPoseCertain": from_test(get_isPoseCertain_test(task)),
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


#######################################################


def post_process(state, plan, replan_obs=True, replan_base=False, look_move=False):
    if plan is None:
        return None
    # TODO: refine actions
    robot = state.task.robot
    commands = []
    uncertain_base = False
    expecting_obs = False
    for i, (name, args) in enumerate(plan):
        if replan_obs and expecting_obs:
            break
        saved_world = WorldSaver()  # StateSaver()
        if name == "picktarget":
            if uncertain_base:
                break
            a, b, p, g, _, c, _ = args
            [t] = c.commands
            attach = Attach(robot, a, g, b)
            new_commands = [t, attach, t.reverse()]
            # expecting_obs=True
        elif name == "pickblock":
            if uncertain_base:
                break
            a, b, p, g, _, c, pb, vs = args
            [t] = c.commands
            attach = Attach(robot, a, g, b)
            new_commands = [t, attach, t.reverse()]
            expecting_obs = True  # To avoid the loop of pick->obs->replan->place->pick->obs(the same loc)
        elif name == "placeblock":
            if uncertain_base:
                break
            a, b, p, g, _, c, pb, _ = args
            [t] = c.commands
            detach = Detach(robot, a, b)
            new_commands = [t, detach, t.reverse()]
            expecting_obs = True
        elif name == "mark-target":
            (o,) = args
            new_commands = [Mark(robot, o)]
        elif name == "observe":  # observation test
            p, bq, vs = args
            obs = Observe(p, bq, vs)
            new_commands = [obs]
            expecting_obs = True
        # elif name == 'observe-specific': # observation test
        #     obj,pobj, vspace = args
        #     obs = Observe_specific(obj,pobj,vspace)
        #     new_commands = [obs]
        #     expecting_obs = True
        else:
            raise ValueError(name)
        saved_world.restore()
        for command in new_commands:
            if isinstance(command, Trajectory) and command.path:
                command.path[-1].assign()
        commands += new_commands
    return commands


#######################################################


def plan_commands(state, viewer=False, teleport=False, profile=False, verbose=True):
    # TODO: could make indices into set of bodies to ensure the same...
    # TODO: populate the bodies here from state and not the real world
    sim_world = connect(use_gui=viewer)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=60,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0, 0.6],
        physicsClientId=sim_world,
    )
    # clone_world(client=sim_world)
    task = state.task
    robot_conf = get_configuration(task.robot)
    robot_pose = get_pose(task.robot)
    # vis_state = task.vis_state
    with ClientSaver(sim_world):  # set client = sim_world on enter
        with HideOutput():
            robot = create_pr2(use_drake=USE_DRAKE_PR2)  # ,client = sim_world)
        set_pose(robot, robot_pose)
        set_configuration(
            robot, robot_conf
        )  # set client = CLIENT on exit(restore CLIENT before with command)
    mapping = clone_world(
        client=sim_world,
        exclude=[task.robot] + list(np.arange(10) + PR2),
        loadmesh=task.body2filename,
    )
    assert all(i1 == i2 for i1, i2 in mapping.items())
    set_client(sim_world)
    saved_world = WorldSaver()  # StateSaver()

    pddlstream_problem = pddlstream_from_state(state, teleport=teleport)
    _, _, _, stream_map, init, goal = pddlstream_problem
    print("Init:", sorted(init, key=lambda f: f[0]))
    if verbose:
        print("Goal:", goal)
        print("Streams:", stream_map.keys())

    stream_info = {
        "test-fully-observed": StreamInfo(eager=True, p_success=0.1),
        # 'test-unblock': StreamInfo(eager=True, p_success=0.1),
        "test-isTarget": StreamInfo(eager=True, p_success=0.1),
        "sample-vis": StreamInfo(eager=True, p_success=1),
        # 'make-obs': StreamInfo(eager=True, p_success=1) #TODO observation test
    }
    hierarchy = [
        ABSTRIPSLayer(pos_pre=["AtBConf"]),
    ]

    pr = cProfile.Profile()
    pr.enable()
    solution = solve_focused(
        pddlstream_problem,
        stream_info=stream_info,
        hierarchy=hierarchy,
        debug=False,
        success_cost=MAX_COST,
        verbose=verbose,
    )
    plan, cost, evaluations = solution
    if MAX_COST <= cost:
        plan = None
    print_solution(solution)
    print("Finite cost:", cost < MAX_COST)
    commands = post_process(state, plan)
    pr.disable()
    if profile:
        pstats.Stats(pr).sort_stats("cumtime").print_stats(10)
    saved_world.restore()
    # apply_commands(state,commands,time_step=0.01)
    disconnect()
    return commands


#######################################################


def add_objects():
    mass = 1
    obj_num = 2  # np.random.randint(2)+4
    table_len, table_h = 0.5, 0.7
    objlist_ = os.listdir(YCB_BANK_DIR)
    objlist = [x for x in objlist_ if x[0] == "0"]
    objlist.sort()
    objlist = objlist  # [:8]

    body2filename = {}
    for i in range(obj_num):
        angle = tuple(np.random.rand(3))
        box_shape = np.random.randint(len(objlist))
        if i == 0:
            box_shape = 7
        box = load_pybullet(f"{YCB_BANK_DIR}/{objlist[box_shape]}/textured.obj", mass=1)
        body2filename[box] = f"{YCB_BANK_DIR}/{objlist[box_shape]}/textured.obj"

        # loc = (0.55+i*.05,0.2,table_h+.05)# if i<=1 else  (0.55+.1,0.2,table_h+.05)
        loc = (
            (np.random.rand(1)[0] - 0.5) * table_len / 2.0 + 0.55,
            (np.random.rand(1)[0] - 0.5) * table_len / 2.0 + 0.2,
            table_h + 0.05,
        )
        set_point(box, loc)
        set_euler(box, angle)

        for _ in range(500):
            p.stepSimulation()

    return body2filename, objlist


import copy

import detectron2
import torch
import torchvision.transforms as transforms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


def init_vision_utils():

    handlers = {}

    """ 6D pose estimator - DenseFusion """
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    num_points = 1000
    num_obj = 21
    bs = 1

    from lib.network import PoseNet, PoseRefineNet

    estimator = PoseNet(num_points=num_points, num_obj=num_obj)
    estimator.cuda()
    estimator.load_state_dict(
        torch.load(
            f"{POSE_DIR}/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth"
        )
    )
    estimator.eval()

    refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
    refiner.cuda()
    refiner.load_state_dict(
        torch.load(
            f"{POSE_DIR}/trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth"
        )
    )
    refiner.eval()

    """ Detection & Segmentation - MaskRCNN """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 23
    cfg.MODEL.WEIGHTS = os.path.join(SEG_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.7  # set the testing threshold for this model
    )
    # cfg.INPUT.MASK_FORMAT
    predictor = DefaultPredictor(cfg)

    """ wrap up """
    handlers["pose6d_init"] = estimator
    handlers["pose6d_refiner"] = refiner
    handlers["mask"] = predictor
    return handlers


def main(time_step=0.01):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-viewer", action="store_true", help="enable the viewer while planning"
    )
    args = parser.parse_args()

    real_world = connect(use_gui=not args.viewer)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-60,
        cameraTargetPosition=[0.5, 1.0, 0.6],
    )

    add_data_path()
    p.setGravity(0, 0, -10)

    task, state = get_problem1(
        localized="rooms", goal_category=7, grasp_types=("top", "side")
    )  # surfaces | rooms

    # TODO parallel adding objects and init vision utils
    body2filename, obj_bank = add_objects()
    task.body2filename = body2filename
    task.obj_bank = obj_bank

    # clone_table & robo
    with HideOutput():
        # from examples.pybullet.utils.pybullet_tools.pr2_utils import arm_conf, set_arm_conf, open_arm, WIDE_LEFT_ARM, get_other_arm
        # arm='left'
        # pr2 = create_pr2(use_drake=USE_DRAKE_PR2)
        # set_arm_conf(pr2, arm, arm_conf(arm, WIDE_LEFT_ARM)) # get_carry_conf(arm, grasp_type))
        # open_arm(pr2, arm) # set_gripper
        # other_arm = get_other_arm(arm)
        # set_arm_conf(pr2, other_arm, arm_conf(other_arm, WIDE_LEFT_ARM)) # TOP_HOLDING_LEFT_ARM | SIDE_HOLDING_LEFT_ARM | REST_LEFT_ARM | WIDE_LEFT_ARM | CENTER_LEFT_ARM
        # global PR2
        # PR2=pr2
        # open_arm(pr2, other_arm) # set gripper
        # set_point(pr2,(0,2.,0))
        w, h = 0.5, 0.55
        table = create_box(w, w * 2, h, color=(0.75, 0.75, 0.75, 1))
        set_point(table, (0.4 + w / 2.0, 2, h / 2))
        global PR2
        PR2 = table
    pose2obj = {}
    obj_list = get_bodies()
    for o in obj_list:
        if o <= 2 or o >= PR2:
            continue
        pose2obj[get_pose(o)[0]] = o
    task.pose2obj = pose2obj
    print(pose2obj)

    robot = task.robot
    assert USE_DRAKE_PR2 == is_drake_pr2(robot)
    step = 0
    vis_handler = init_vision_utils()
    task.vis_handler = vis_handler

    while True:
        step += 1
        print("\n" + 50 * "-")
        print("Block list: ", state.block_list)
        with ClientSaver():
            commands = plan_commands(state, viewer=args.viewer)
        print()
        if commands is None:
            print("Failure!")
            break
        if not commands:
            print("Success!")
            break
        apply_commands(state, commands, time_step=time_step)
        wait_for_user()

    print(f"{len(state.registered.keys())} objects detected.")
    wait_for_user()
    disconnect()


if __name__ == "__main__":
    main()
