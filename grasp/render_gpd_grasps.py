#!/usr/bin/env python3

from __future__ import print_function

import copy
import os
import random
import sys
import time
import warnings

import numpy as np
import pybullet as p

warnings.filterwarnings("ignore")  # , category=DeprecationWarning)

np.set_printoptions(
    precision=3, threshold=3, edgeitems=1, suppress=True
)  # , linewidth=1000)

MODEL_PATH = "./models"
LTAMP_PR2 = os.path.join(MODEL_PATH, "ltamp/pr2_description/pr2.urdf")
SUGAR_PATH = os.path.join(MODEL_PATH, "srl/ycb/010_potted_meat_can/textured.obj")
GRIPPER_TEMPLATE = os.path.join(MODEL_PATH, "panda_gripper.obj")

# NOTE(caelan): must come before other imports
sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
        #'pddlstream/examples/pybullet/utils',
    ]
)

from collections import namedtuple

import sklearn
from pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics
from pybullet_tools.pr2_primitives import (
    Commands,
    Grasp,
    State,
    apply_commands,
    create_trajectory,
)
from pybullet_tools.pr2_utils import (
    ARM_NAMES,
    LEFT_ARM,
    PR2_GROUPS,
    PR2_TOOL_FRAMES,
    TOP_HOLDING_LEFT_ARM,
    arm_conf,
    get_arm_joints,
    get_gripper_link,
    joints_from_names,
    open_arm,
)
from pybullet_tools.utils import (
    PI,
    TEMP_DIR,
    BodySaver,
    Point,
    Pose,
    add_fixed_constraint,
    body_collision,
    connect,
    disconnect,
    draw_pose,
    get_joint_positions,
    get_link_pose,
    get_max_limits,
    get_movable_joints,
    get_pose,
    get_unit_vector,
    image_from_segmented,
    invert,
    is_placement,
    link_from_name,
    load_pybullet,
    multiply,
    plan_direct_joint_motion,
    plan_joint_motion,
    randomize,
    remove_body,
    remove_handles,
    save_image,
    set_joint_positions,
    set_pose,
    sub_inverse_kinematics,
    unit_quat,
    wait_for_duration,
    wait_if_gui,
)

connect(
    use_gui=True
)  # Must be before run_grasps and run_estimator (something about main?)

from environments import create_default_env
from run_grasps import Pose2D, place_object

from grasp.utils import (
    GPD_GRIPPER_ADJUSTMENT,
    gpd_predict_grasps,
    graspnet_predict_grasps,
)
from open_world.estimation.observation import iterate_point_cloud
from open_world.simulation.entities import RealWorld
from open_world.simulation.environment import create_ycb
from open_world.simulation.lis import YCB_MASSES
from run_estimator import create_parser

#######################################################

LabeledPoint = namedtuple("LabeledPoint", ["point", "color", "body"])


def set_group_positions(pr2, group_positions):
    for group, positions in group_positions.items():
        joints = joints_from_names(pr2, PR2_GROUPS[group])
        assert len(joints) == len(positions)
        set_joint_positions(pr2, joints, positions)


#######################################################


def convexify(labeled_points, points_to_add=10000):
    new_labeled_points = copy.deepcopy(labeled_points)
    for _ in range(points_to_add):
        pntidx = random.sample(labeled_points, 2)
        t = random.uniform(0, 1)
        new_point = tuple(
            [pntidx[0].point[i] * t + pntidx[1].point[i] * (1 - t) for i in range(3)]
        )
        new_labeled_points.append(
            LabeledPoint(new_point, pntidx[0].color, pntidx[0].body)
        )
    return new_labeled_points


def apply_physical_commands(state, commands, time_step=None, **kwargs):
    for i, command in enumerate(commands):
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            else:
                wait_for_duration(time_step)
                p.stepSimulation()


def create_world(robot, movable=[], fixed=[], surfaces=[], materials={}, concave=False):
    obstacles = sorted(set(fixed) | set(surfaces))
    return RealWorld(
        robot,
        movable=movable,
        fixed=obstacles,
        detectable=movable,
        known=obstacles,
        surfaces=surfaces,
        materials=materials,
        concave=concave,
    )


#######################################################


def main():

    arms = [LEFT_ARM]
    np.set_printoptions(
        precision=3, threshold=3, suppress=True
    )  # , edgeitems=1) #, linewidth=1000)

    parser = create_parser()
    parser.add_argument(
        "-y", "--ycb", default="power_drill", choices=sorted(YCB_MASSES), help=""
    )
    parser.add_argument("-sv", "--save_video", action="store_true", help="")
    args = parser.parse_args()

    # connect(use_gui=args.viewer)

    robot, table, obstacles = create_default_env(arms=arms)

    side = "left"
    arm_group, gripper_group, tool_name = robot.manipulators[side]
    tool_link = link_from_name(robot, tool_name)
    # robot.dump()
    # parent_from_tool = robot.get_parent_from_tool(side)
    # print(parent_from_tool) # palm_from_tool

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj = place_object(
        create_ycb(args.ycb, use_concave=True), table, Pose2D(yaw=PI / 4)
    )
    obj_pose = get_pose(obj)

    # take picture
    camera_image = robot.cameras[0].get_image()
    rgb_image, depth_image, seg_image, camera_pose, camera_matrix = camera_image

    save_image(os.path.join(TEMP_DIR, "rgb.png"), rgb_image)  # [0, 255]
    save_image(os.path.join(TEMP_DIR, "depth.png"), depth_image)  # [0, 1]
    if seg_image is not None:
        segmented_image = image_from_segmented(seg_image, color_from_body=None)
        save_image(os.path.join(TEMP_DIR, "segmented.png"), segmented_image)  # [0, 1]

    all_points = [lp for lp in iterate_point_cloud(camera_image)]
    points = np.array([point.point for point in all_points if point.label[0] == obj])

    # TODO: simplify meshes

    if args.grasp_mode == "graspnet":
        grasps, scores = graspnet_predict_grasps(points, camera_pose)
    else:
        grasps, scores = gpd_predict_grasps(points, camera_pose, use_tool=True)

    #######################################################

    # Load in the URDF as an object
    gripper_path = "models/ltamp/pr2_description/pr2_l_gripper.urdf"
    # gripper_path = WSG_GRIPPER
    # gripper_path = KUKA_IIWA_URDF

    gripper = load_pybullet(
        gripper_path, fixed_base=True
    )  # root is l_gripper_palm_joint
    gripper_joints = get_movable_joints(gripper)
    open_conf = get_max_limits(gripper, gripper_joints)
    set_joint_positions(gripper, gripper_joints, open_conf)

    # for link_name in ['l_gripper_palm_link', 'l_gripper_tool_frame']:
    tool_from_gripper = multiply(
        invert(get_link_pose(gripper, link_from_name(gripper, "l_gripper_tool_frame"))),
        get_pose(gripper),
    )

    # gripper2 = load_pybullet(gripper_path, fixed_base=True)
    # gripper_joints2 = get_movable_joints(gripper2)
    # set_joint_positions(gripper2, gripper_joints2, open_conf)

    templates = []
    grasp_transforms = []
    orig_grasp_index = []
    grasps = randomize(grasps)
    for gi, world_from_tool in enumerate(grasps):

        # This rotation appears to give the correct gripper orientation for around half the grasps.
        # Possibly the gripper needs to be flipped rather than rotated or some combination.
        # Either way, the gripper also needs to be pushed backward since the fingers are shorted than the gpd specificed gripper
        # pose = multiply(get_pose(obj), pose)
        gripper_pose = multiply(world_from_tool, tool_from_gripper)
        set_pose(gripper, gripper_pose)

        if any(body_collision(gripper, obst) for obst in [table, obj]):
            continue

        tool_pose = get_link_pose(
            gripper, link_from_name(gripper, "l_gripper_tool_frame")
        )
        handles = draw_pose(tool_pose)
        wait_if_gui()
        remove_handles(handles)

    sys.exit(1)

    #######################################################

    grasp_index = 0
    # grasp_index = 57
    # grasp_index = random.randint(0, len(grasp_transforms)-1)
    remaining = None
    for ti, template in enumerate(templates):
        if ti != grasp_index:
            remove_body(template)
        else:
            remaining = template

    # Get the agent to actually grasp the object with the grasp transpose

    arm = ARM_NAMES[0]
    GRASP_LENGTH = 0.03
    APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
    approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0])

    grasp_targets = [
        ((g[0][0], g[0][1], g[0][2] + 0.1), (g[1][0], g[1][1], g[1][2], g[1][3]))
        for g in [grasp_transforms[grasp_index]]
    ]
    grasps = []
    grasps.extend(
        Grasp(
            "top",
            obj,
            g,
            multiply((approach_vector, unit_quat()), g),
            TOP_HOLDING_LEFT_ARM,
        )
        for g in [grasp_transforms[grasp_index]]
    )

    # print("GRASP: " + str())

    arm_link = get_gripper_link(robot, arm)
    arm_joints = get_arm_joints(robot, arm)

    pose = get_pose(obj)
    obstacles = []
    approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}

    grasp = grasps[0]
    gripper_pose = grasp.value  # w_f_g = w_f_o * (g_f_o)^-1
    approach_pose = grasp.approach
    default_conf = arm_conf(arm, grasp.carry)
    # sample_fn = get_sample_fn(robot, arm_joints)
    # base_conf.assign()
    custom_limits = {}

    open_arm(robot, arm)
    set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()

    print(robot, arm, gripper_pose)
    grasp_conf = pr2_inverse_kinematics(
        robot, arm, gripper_pose, custom_limits=custom_limits
    )  # , upper_limits=USE_CURRENT)

    approach_conf = sub_inverse_kinematics(
        robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits
    )
    approach_conf = get_joint_positions(robot, arm_joints)

    resolutions = 0.01 ** np.ones(len(arm_joints))
    grasp_path = plan_direct_joint_motion(
        robot,
        arm_joints,
        grasp_conf,
        obstacles=approach_obstacles,
        self_collisions=False,
        resolutions=resolutions / 2.0,
        attachments=[],
    )
    if grasp_path is None:
        print("Grasp path failure")
        return None
    set_joint_positions(robot, arm_joints, default_conf)
    approach_path = plan_joint_motion(
        robot,
        arm_joints,
        approach_conf,
        obstacles=obstacles,
        self_collisions=False,
        resolutions=resolutions,
        restarts=2,
        iterations=25,
        smooth=25,
    )
    if approach_path is None:
        print("Approach path failure")
        return None

    # TARGET POSITION
    target_grasps = []
    target_grasps.extend(
        Grasp(
            "top",
            obj,
            g,
            multiply((approach_vector, unit_quat()), g),
            TOP_HOLDING_LEFT_ARM,
        )
        for g in grasp_targets
    )

    target_grasp = target_grasps[0]
    target_gripper_pose = target_grasp.value  # w_f_g = w_f_o * (g_f_o)^-1
    target_approach_pose = target_grasp.approach

    set_joint_positions(robot, arm_joints, grasp_conf)  # default_conf | sample_fn()

    target_grasp_conf = pr2_inverse_kinematics(
        robot, arm, target_gripper_pose, custom_limits=custom_limits
    )  # , upper_limits=USE_CURRENT)
    target_approach_conf = sub_inverse_kinematics(
        robot,
        arm_joints[0],
        arm_link,
        target_approach_pose,
        custom_limits=custom_limits,
    )
    target_approach_conf = get_joint_positions(robot, arm_joints)

    resolutions = 0.01 ** np.ones(len(arm_joints))

    target_grasp_path = plan_direct_joint_motion(
        robot,
        arm_joints,
        target_grasp_conf,
        obstacles=approach_obstacles,
        self_collisions=False,
        resolutions=resolutions / 2.0,
        attachments=[],
    )

    set_joint_positions(robot, arm_joints, grasp_conf)
    target_approach_path = plan_joint_motion(
        robot,
        arm_joints,
        target_approach_conf,
        obstacles=obstacles,
        self_collisions=False,
        resolutions=resolutions,
        restarts=2,
        iterations=25,
        smooth=25,
    )
    if approach_path is None:
        print("Approach path failure")
        return None

    path = approach_path + grasp_path
    pre_mt = create_trajectory(robot, arm_joints, path)
    print("pre_mt: " + str(path))
    pre_grasp_cmd = Commands(State(), savers=[BodySaver(robot)], commands=[pre_mt])

    target_path = target_approach_path
    post_mt = create_trajectory(robot, arm_joints, target_path)
    print("postmt: " + str(target_grasp_path))
    post_grasp_cmd = Commands(State(), savers=[BodySaver(robot)], commands=[post_mt])
    # pdb.set_trace()
    remove_body(remaining)
    apply_commands(State(), pre_grasp_cmd.commands, time_step=0.01)
    link = link_from_name(robot, PR2_TOOL_FRAMES.get(arm, arm))
    add_fixed_constraint(obj, robot, link)

    apply_physical_commands(State(), post_grasp_cmd.commands, time_step=0.01)
    for _ in range(500):
        p.stepSimulation()
        time.sleep(0.01)
    wait_if_gui()

    disconnect()


if __name__ == "__main__":
    main()
