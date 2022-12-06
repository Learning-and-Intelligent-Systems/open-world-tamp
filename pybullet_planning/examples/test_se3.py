#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from pybullet_tools.utils import (
    BLUE,
    GREEN,
    RED,
    SE3,
    add_line,
    connect,
    create_box,
    create_flying_body,
    create_shape,
    disconnect,
    dump_body,
    euler_from_quat,
    get_cylinder_geometry,
    get_link_pose,
    get_links,
    get_movable_joints,
    intrinsic_euler_from_quat,
    plan_joint_motion,
    set_camera_pose,
    set_joint_positions,
    wait_for_user,
)

SIZE = 1.0

CUSTOM_LIMITS = {
    "x": (-SIZE, SIZE),
    "y": (-SIZE, SIZE),
    "z": (-SIZE, SIZE),
}


def main(group=SE3):
    connect(use_gui=True)
    set_camera_pose(camera_point=SIZE * np.array([1.0, -1.0, 1.0]))
    # TODO: can also create all links and fix some joints
    # TODO: SE(3) motion planner (like my SE(3) one) where some dimensions are fixed

    obstacle = create_box(w=SIZE, l=SIZE, h=SIZE, color=RED)
    # robot = create_cylinder(radius=0.025, height=0.1, color=BLUE)
    obstacles = [obstacle]

    collision_id, visual_id = create_shape(
        get_cylinder_geometry(radius=0.025, height=0.1), color=BLUE
    )
    robot = create_flying_body(group, collision_id, visual_id)

    body_link = get_links(robot)[-1]
    joints = get_movable_joints(robot)
    joint_from_group = dict(zip(group, joints))
    print(joint_from_group)
    # print(get_aabb(robot, body_link))
    dump_body(robot, fixed=False)
    custom_limits = {joint_from_group[j]: l for j, l in CUSTOM_LIMITS.items()}

    # sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    # for i in range(10):
    #     conf = sample_fn()
    #     set_joint_positions(robot, joints, conf)
    #     wait_for_user('Iteration: {}'.format(i))
    # return

    initial_point = SIZE * np.array([-1.0, -1.0, 0])
    # initial_point = -1.*np.ones(3)
    final_point = -initial_point
    initial_euler = np.zeros(3)
    add_line(initial_point, final_point, color=GREEN)

    initial_conf = np.concatenate([initial_point, initial_euler])
    final_conf = np.concatenate([final_point, initial_euler])

    set_joint_positions(robot, joints, initial_conf)
    # print(initial_point, get_link_pose(robot, body_link))
    # set_pose(robot, Pose(point=-1.*np.ones(3)))

    # TODO: sample orientation uniformly at random
    # http://planning.cs.uiuc.edu/node198.html
    # from pybullet_tools.transformations import random_quaternion
    path = plan_joint_motion(
        robot,
        joints,
        final_conf,
        obstacles=obstacles,
        self_collisions=False,
        custom_limits=custom_limits,
    )
    if path is None:
        disconnect()
        return

    for i, conf in enumerate(path):
        set_joint_positions(robot, joints, conf)
        point, quat = get_link_pose(robot, body_link)
        # euler = euler_from_quat(quat)
        euler = intrinsic_euler_from_quat(quat)
        print(conf)
        print(point, euler)
        wait_for_user("Step: {}/{}".format(i, len(path)))

    wait_for_user("Finish?")
    disconnect()


if __name__ == "__main__":
    main()
