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
import random
from pybullet_tools.utils import (
    RGBA,
    TAN,
    CameraImage,
    Point,
    Pose,
    create_box,
    invert,
    pixel_from_point,
    set_pose,
    tform_point,
    get_movable_joints,
    get_joint_positions,
    get_joint_names,
    set_joint_positions
)
from movo.movo_worlds import create_world
from open_world.planning.samplers import compute_gripper_path
from open_world.estimation.observation import (
    save_camera_images,
)


from open_world.planning.primitives import (
    GroupConf,
    RelativePose,
)
from open_world.planning.streams import (
    get_grasp_gen_fn,
    get_plan_mobile_pick_fn,
    get_plan_motion_fn,
)
from open_world.simulation.entities import Object
from open_world.simulation.environment import (
    create_pillar
)
from open_world.simulation.policy import (
    CameraImage,
    link_seg_from_gt,
)
from open_world.simulation.tasks import GOALS

from run_planner import (
    create_parser,
    robot_entities,
    setup_robot_pybullet,
)
from open_world.planning.drake.drake_controller import drake_ik

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


def sample_state_particle(robot):
    arms = args.arms  # ARM_NAMES
    floor_size = 6
    floor = create_pillar(width=floor_size, length=floor_size, color=TAN)

    box_side = 0.05
    box_mass = 0.2
    height = box_side * 8
    red_box = Object(
        create_box(
            w=box_side,
            l=box_side,
            h=height,
            color=RGBA(219 / 256.0, 50 / 256.0, 54 / 256.0, 1.0),
            mass=box_mass,
        )
    )

    box_x = random.uniform(-floor_size // 2, floor_size // 2)
    box_y = random.uniform(-floor_size // 2, floor_size // 2)
    set_pose(red_box, Pose(point=Point(x=box_x, y=box_y, z=height / 2.0)))

    real_world = create_world(robot, movable=[red_box], fixed=[], surfaces=[floor])
    return real_world


if __name__ == "__main__":
    # Parse the args
    problem_from_name = {fn.__name__: fn for fn in GOALS}
    parser = create_parser()
    args = parser.parse_args()

    # Create the robot
    robot_body, _ = setup_robot_pybullet(args)
    robot = robot_entities[args.robot](robot_body, args=args)
    reset_robot(robot)

    # Create the task
    if args.goal not in problem_from_name:
        raise ValueError(args.goal)
    problem_fn = problem_from_name[args.goal]
    task = problem_fn(args)

    real_world = sample_state_particle(robot)
    camera_image = get_camera_image(args, robot)
    resolutions = 0.2 * np.ones(3)
    surface_origin = Pose(Point(z=0.01))
    height = 0.1
    # surface_aabb = AABB(lower=(-3, -3, 0.05), upper=(3, 3, 0 + height))  # TODO: buffer
    # vg = VoxelGrid(
    #     resolutions, world_from_grid=surface_origin, aabb=surface_aabb, color=BLUE
    # )
    # for voxel in vg.voxels_from_aabb(surface_aabb):
    #     vg.set_occupied(voxel)
    # update_visibility(camera_image, visibility_grid)

    motion_planner = get_plan_motion_fn(robot, environment=[])
    mobile_pick_planner = get_plan_mobile_pick_fn(robot)
    grasp_finder = get_grasp_gen_fn(robot, [], grasp_mode='top')

    init_confs = {group: GroupConf(robot, group, important=True) for group in robot.groups}
    arm = "right_arm"
    obj = real_world.movable[0]
    pose = RelativePose(obj)
    grasp,  = next(grasp_finder(arm, obj))


    from open_world.planning.drake.drake_controller import drake_ik

    pose.assign()
    gripper_path = compute_gripper_path(pose, grasp)
    gripper_pose = gripper_path[0]

    movable_joints = get_movable_joints(robot)
    q0 = get_joint_positions(robot, movable_joints)
    movable_joint_names = get_joint_names(robot, movable_joints)
    positions = drake_ik(gripper_pose[0], gripper_pose[1], movable_joint_names, q0)

    print(positions)
    print("Setting joint positions")
    set_joint_positions(robot, movable_joints, positions)

    while True:
        time.sleep(0.01)
        p.setGravity(0,0,0)
        set_joint_positions(robot, movable_joints, positions)

