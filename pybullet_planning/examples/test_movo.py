#!/usr/bin/env python

from __future__ import print_function

import time

import numpy as np
import pybullet as p
from examples.test_franka import test_retraction
from pybullet_tools.ikfast.ikfast import check_ik_solver, get_ik_joints
from pybullet_tools.movo_constants import (
    ARMS,
    BASE_JOINTS,
    MOVO_COLOR,
    MOVO_INFOS,
    MOVO_URDF,
    RIGHT,
    TOOL_LINK,
    get_arm_joints,
    get_closed_positions,
    get_gripper_joints,
    get_open_positions,
)
from pybullet_tools.pr2_utils import close_until_collision, get_side_grasps
from pybullet_tools.utils import (
    RED,
    HideOutput,
    LockRenderer,
    add_data_path,
    assign_link_colors,
    connect,
    create_box,
    disconnect,
    draw_base_limits,
    draw_pose,
    dump_body,
    elapsed_time,
    get_extend_fn,
    get_joint_names,
    get_link_pose,
    get_links,
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    link_from_name,
    load_model,
    multiply,
    remove_handles,
    set_all_color,
    set_color,
    set_joint_positions,
    set_pose,
    unit_pose,
    wait_if_gui,
)


def test_close_gripper(robot, arm):
    gripper_joints = get_gripper_joints(robot, arm)
    extend_fn = get_extend_fn(robot, gripper_joints)
    for positions in extend_fn(
        get_open_positions(robot, arm), get_closed_positions(robot, arm)
    ):
        set_joint_positions(robot, gripper_joints, positions)
        print(positions)
        wait_if_gui("Continue?")


def test_grasps(robot, block):
    for arm in ARMS:
        gripper_joints = get_gripper_joints(robot, arm)
        tool_link = link_from_name(robot, TOOL_LINK.format(arm))
        tool_pose = get_link_pose(robot, tool_link)
        # handles = draw_pose(tool_pose)
        # grasps = get_top_grasps(block, under=True, tool_pose=unit_pose())
        grasps = get_side_grasps(block, under=True, tool_pose=unit_pose())
        for i, grasp_pose in enumerate(grasps):
            block_pose = multiply(tool_pose, grasp_pose)
            set_pose(block, block_pose)
            close_until_collision(
                robot,
                gripper_joints,
                bodies=[block],
                open_conf=get_open_positions(robot, arm),
                closed_conf=get_closed_positions(robot, arm),
            )
            handles = draw_pose(block_pose)
            wait_if_gui("Grasp {}".format(i))
            remove_handles(handles)


#####################################


def main(num_iterations=10):
    # The URDF loader seems robust to package:// and slightly wrong relative paths?
    connect(use_gui=True)
    add_data_path()
    plane = p.loadURDF("plane.urdf")
    side = 0.05
    block = create_box(w=side, l=side, h=side, color=RED)

    start_time = time.time()
    with LockRenderer():
        with HideOutput():
            # TODO: MOVO must be loaded last
            robot = load_model(MOVO_URDF, fixed_base=True)
        # set_all_color(robot, color=MOVO_COLOR)
        assign_link_colors(robot)
        base_joints = joints_from_names(robot, BASE_JOINTS)
        draw_base_limits(
            (get_min_limits(robot, base_joints), get_max_limits(robot, base_joints)),
            z=1e-2,
        )
    print("Load time: {:.3f}".format(elapsed_time(start_time)))

    dump_body(robot)
    # print(get_colliding(robot))
    # for arm in ARMS:
    #    test_close_gripper(robot, arm)
    # test_grasps(robot, block)

    arm = RIGHT
    tool_link = link_from_name(robot, TOOL_LINK.format(arm))

    # joint_names = HEAD_JOINTS
    # joints = joints_from_names(robot, joint_names)
    joints = base_joints + get_arm_joints(robot, arm)
    # joints = get_movable_joints(robot)
    print("Joints:", get_joint_names(robot, joints))

    ik_info = MOVO_INFOS[arm]
    check_ik_solver(ik_info)

    ik_joints = get_ik_joints(robot, ik_info, tool_link)
    # fixed_joints = []
    fixed_joints = ik_joints[:1]
    # fixed_joints = ik_joints

    wait_if_gui("Start?")
    sample_fn = get_sample_fn(robot, joints)
    handles = []
    for i in range(num_iterations):
        conf = sample_fn()
        print(
            "Iteration: {}/{} | Conf: {}".format(i + 1, num_iterations, np.array(conf))
        )
        set_joint_positions(robot, joints, conf)
        tool_pose = get_link_pose(robot, tool_link)
        remove_handles(handles)
        handles = draw_pose(tool_pose)
        wait_if_gui()

        # conf = next(ikfast_inverse_kinematics(robot, MOVO_INFOS[arm], tool_link, tool_pose,
        #                                      fixed_joints=fixed_joints, max_time=0.1), None)
        # if conf is not None:
        #    set_joint_positions(robot, ik_joints, conf)
        # wait_if_gui()
        test_retraction(
            robot,
            ik_info,
            tool_link,
            fixed_joints=fixed_joints,
            max_time=0.05,
            max_candidates=100,
        )
    disconnect()


if __name__ == "__main__":
    main()
