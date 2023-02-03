#!/usr/bin/env python3

from __future__ import print_function

import sys
import time
import warnings

import numpy as np
import pybullet as p

warnings.filterwarnings("ignore")
sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
    ]
)

import pybullet as p
from pybullet_tools.utils import (
    CameraImage,
    Point,
    Pose,
    link_from_name,
    invert,
    pixel_from_point,
    tform_point,
    get_movable_joints,
    get_joint_positions,
    get_joint_limits,
    get_joint_names,
)


from open_world.planning.samplers import compute_gripper_path, plan_workspace_motion
from open_world.estimation.observation import (
    save_camera_images,
)
from open_world.planning.planner import (
    iterate_sequence,
)
from open_world.planning.primitives import (
    GroupConf,
    RelativePose,
    Sequence,
    WorldState,
    GroupTrajectory
)
from open_world.planning.streams import (
    get_grasp_gen_fn,
    get_plan_mobile_pick_fn,
    get_plan_motion_fn,
)

from open_world.simulation.policy import (
    CameraImage,
    link_seg_from_gt,
)
from open_world.simulation.tasks import  GOALS
from run_planner import (
    create_parser,
    robot_entities,
    robot_simulated_worlds,
    setup_robot_pybullet,
)

from open_world.planning.drake.drake_planner import drake_motion_planning

from pybullet_tools.ikfast.ikfast import (
    get_ik_joints,
)

SAVE_DIR = "temp_graphs/"


def get_camera_image(args, robot):
    [camera] = robot.cameras
    camera_image = camera.get_image(segment_links=True)
    rgb, depth, predicted_seg, camera_pose, camera_matrix = (
        camera_image.rgbPixels,
        camera_image.depthPixels,
        camera_image.segmentationMaskBuffer,
        camera_image.camera_pose,
        camera_image.camera_matrix,
    )
    camera_image = CameraImage(
        rgb, depth, link_seg_from_gt(predicted_seg), camera_pose, camera_matrix
    )
    save_camera_images(camera_image, directory=SAVE_DIR)

    return camera_image


def update_visibility(camera_image, visibility_grid, **kwargs):
    surface_aabb = visibility_grid.aabb
    camera_pose, camera_matrix = camera_image[-2:]
    grid = visibility_grid
    for voxel in grid.voxels_from_aabb(surface_aabb):
        center_world = grid.to_world(grid.center_from_voxel(voxel))
        center_camera = tform_point(invert(camera_pose), center_world)
        distance = center_camera[2]
        # if not (0 <= distance < max_depth):
        #    continue
        pixel = pixel_from_point(camera_matrix, center_camera)
        if pixel is not None:
            # TODO: local filter
            r, c = pixel
            depth = camera_image.depthPixels[r, c]
            # if distance > depth:
            #     grid.set_occupied(voxel)
            #     # grid.add_point(center_world) # TODO: check pixels within voxel bounding box
            if distance <= depth:
                grid.set_free(voxel)
    return grid


def reset_robot(robot):
    conf = robot.get_default_conf()
    for group, positions in conf.items():
        robot.set_group_positions(group, positions)


if __name__ == "__main__":
    # Parse the args
    problem_from_name = {fn.__name__: fn for fn in GOALS}
    parser = create_parser()
    args = parser.parse_args()

    # Create the robot
    robot_body, client = setup_robot_pybullet(args)
    robot = robot_entities[args.robot](robot_body, args=args)
    reset_robot(robot)

    # Create the task
    if args.goal not in problem_from_name:
        raise ValueError(args.goal)
    problem_fn = problem_from_name[args.goal]
    task = problem_fn(args)

    goal_positions = None
    real_world = robot_simulated_worlds[args.robot]("problem0", robot, args, client=client)

    camera_image = get_camera_image(args, robot)
    resolutions = 0.2 * np.ones(3)
    surface_origin = Pose(Point(z=0.01))
    height = 0.1

    motion_planner = get_plan_motion_fn(robot, environment=[])
    mobile_pick_planner = get_plan_mobile_pick_fn(robot)
    grasp_finder = get_grasp_gen_fn(robot, [], grasp_mode='mesh')

    # Arm joints
    arm = "right_arm"

    ik_info = robot.ik_info["right"]
    arm_group, _, tool_name = robot.manipulators["right"]
    tool_link = link_from_name(robot, tool_name, client=client)
    ik_joints = get_ik_joints(robot, ik_info, tool_link)  # Arm + torso
    fixed_joints = set(ik_joints) - set(robot.get_group_joints(arm_group))  # Torso only
    arm_joints = [j for j in ik_joints if j not in fixed_joints]  # Arm only

    movable_joints = get_movable_joints(robot)
    q0 = get_joint_positions(robot, movable_joints)

    arm_joint_names = get_joint_names(robot, arm_joints)
    lower, upper = zip(*[get_joint_limits(robot, j) for j in movable_joints])
    
    while(True):
        
        init_confs = {group: GroupConf(robot, group, important=True) for group in robot.groups}
        obj = real_world.movable[0]
        pose = RelativePose(obj)
        grasp,  = next(grasp_finder(arm, obj))

        pose.assign()
        gripper_path = compute_gripper_path(pose, grasp)
        gripper_pose = gripper_path[0]
        

        movable_joint_names = get_joint_names(robot, movable_joints)
        st = time.time()

        goal_positions = plan_workspace_motion(robot, "right", [gripper_pose])
        # goal_positions = drake_ik(gripper_pose[0], gripper_pose[1], movable_joint_names, q0)
        # print("Drakeik")
        # print(time.time()-st)
        
        if(goal_positions is None):
            continue
    
        # Restrict to only planning with the arm
        goal_dict = {k:v for k, v in zip(arm_joint_names, goal_positions[0])} 
        qg = []
        lower = []
        upper = []
        movable_joint_idxs = []
        for ji, (movable_joint, movable_joint_name, q0j) in enumerate(zip(movable_joints, movable_joint_names, q0)):
            if(movable_joint_name in arm_joint_names):
                movable_joint_idxs.append(ji)
                qg.append(goal_dict[movable_joint_name])
                l, u = get_joint_limits(robot, movable_joint)
                lower.append(l)
                upper.append(u)
            else:
                qg.append(q0j)
                lower.append(q0j)
                upper.append(q0j)


        motion_plan = drake_motion_planning(q0, qg, movable_joint_names, lower, upper, obstacles=[])
        if(motion_plan is None):
            continue
        else:
            motion_plan = [j[movable_joint_idxs] for j in motion_plan]
            break

    # Construct an arm trajectory from the output plan
    sequence = Sequence(
        commands=[
            GroupTrajectory(robot, arm, motion_plan, client=robot.client),
        ],
        name="move-{}".format(arm),
    )

    p.removeAllUserDebugItems()
    robot.remove_components()
    reset_robot(robot)
    state = WorldState()
    state.assign()
    iterate_sequence(state, sequence)


    while(True):
        time.sleep(0.1)