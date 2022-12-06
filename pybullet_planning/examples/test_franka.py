#!/usr/bin/env python

from __future__ import print_function

import pybullet as p
from pybullet_tools.ikfast.panda.ik import FRANKA_URDF, PANDA_INFO
from pybullet_tools.ikfast.ikfast import (
    check_ik_solver,
    either_inverse_kinematics,
    get_ik_joints,
)
from pybullet_tools.utils import (
    BLUE,
    INF,
    HideOutput,
    LockRenderer,
    Point,
    Pose,
    add_data_path,
    add_line,
    assign_link_colors,
    connect,
    disconnect,
    draw_pose,
    dump_body,
    get_joint_name,
    get_link_pose,
    get_movable_joints,
    get_sample_fn,
    interpolate_poses,
    link_from_name,
    load_pybullet,
    multiply,
    point_from_pose,
    remove_handles,
    set_camera_pose,
    set_joint_positions,
    wait_for_user,
)


def test_retraction(robot, info, tool_link, distance=0.1, **kwargs):
    ik_joints = get_ik_joints(robot, info, tool_link)
    start_pose = get_link_pose(robot, tool_link)
    end_pose = multiply(start_pose, Pose(Point(z=-distance)))
    handles = [
        add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)
    ]
    # handles.extend(draw_pose(start_pose))
    # handles.extend(draw_pose(end_pose))
    path = []
    pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
    for i, pose in enumerate(pose_path):
        print("Waypoint: {}/{}".format(i + 1, len(pose_path)))
        handles.extend(draw_pose(pose))
        conf = next(
            either_inverse_kinematics(robot, info, tool_link, pose, **kwargs), None
        )
        if conf is None:
            print("Failure!")
            path = None
            wait_for_user()
            break
        set_joint_positions(robot, ik_joints, conf)
        path.append(conf)
        wait_for_user()
        # for conf in islice(ikfast_inverse_kinematics(robot, info, tool_link, pose, max_attempts=INF, max_distance=0.5), 1):
        #    set_joint_positions(robot, joints[:len(conf)], conf)
        #    wait_for_user()
    remove_handles(handles)
    return path


def test_ik(robot, info, tool_link, tool_pose):
    draw_pose(tool_pose)
    # TODO: sort by one joint angle
    # TODO: prune based on proximity
    ik_joints = get_ik_joints(robot, info, tool_link)
    for conf in either_inverse_kinematics(
        robot,
        info,
        tool_link,
        tool_pose,
        use_pybullet=False,
        max_distance=INF,
        max_time=10,
        max_candidates=INF,
    ):
        # TODO: profile
        set_joint_positions(robot, ik_joints, conf)
        wait_for_user()


#####################################


def main():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.0)
    set_camera_pose(camera_point=[1, -1, 1])

    plane = p.loadURDF("plane.urdf")
    with LockRenderer():
        with HideOutput(True):
            robot = load_pybullet(FRANKA_URDF, fixed_base=True)
            assign_link_colors(robot, max_colors=3, s=0.5, v=1.0)
            # set_all_color(robot, GREEN)
    obstacles = [plane]  # TODO: collisions with the ground

    dump_body(robot)
    print("Start?")
    wait_for_user()

    info = PANDA_INFO
    tool_link = link_from_name(robot, "panda_hand")
    draw_pose(Pose(), parent=robot, parent_link=tool_link)
    joints = get_movable_joints(robot)
    print("Joints", [get_joint_name(robot, joint) for joint in joints])
    check_ik_solver(info)

    sample_fn = get_sample_fn(robot, joints)
    for i in range(10):
        print("Iteration:", i)
        conf = sample_fn()
        set_joint_positions(robot, joints, conf)
        wait_for_user()
        # test_ik(robot, info, tool_link, get_link_pose(robot, tool_link))
        test_retraction(
            robot,
            info,
            tool_link,
            use_pybullet=False,
            max_distance=0.1,
            max_time=0.05,
            max_candidates=100,
        )
    disconnect()


if __name__ == "__main__":
    main()
