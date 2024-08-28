from __future__ import print_function

import random

import numpy as np
import pybullet as p

import owt.pb_utils as pbu
from owt.planning.primitives import GroupConf
from owt.simulation.lis import USING_ROS

COLLISION_DISTANCE = 5e-3  # Distance from fixed obstacles
MOVABLE_DISTANCE = COLLISION_DISTANCE
EPSILON = 1e-3
SELF_COLLISIONS = True  # TODO: check self collisions
MAX_IK_TIME = 0.05 if USING_ROS else 0.01
MAX_IK_DISTANCE = np.inf if USING_ROS else np.inf  # math.radians(30)
MAX_TOOL_DISTANCE = np.inf
DISABLE_ALL_COLLISIONS = True


def get_closest_distance(robot, arm_joints, parent_link, tool_link, gripper_pose, obj):
    reach_pose = (pbu.point_from_pose(gripper_pose), None)
    sample_fn = pbu.get_sample_fn(robot, arm_joints)
    pbu.set_joint_positions(robot, arm_joints, sample_fn())
    pbu.inverse_kinematics(robot, arm_joints[0], tool_link, reach_pose)
    collision_infos = pbu.get_closest_points(
        robot, obj, link1=parent_link, max_distance=np.inf
    )
    for collision_info in collision_infos:
        pbu.draw_collision_info(collision_info)

    return min(
        [np.inf]
        + [collision_info.contactDistance for collision_info in collision_infos]
    )


#######################################################


def compute_gripper_path(pose, grasp, pos_step_size=0.02):
    grasp_pose = pbu.multiply(pose.get_pose(), pbu.invert(grasp.grasp))
    pregrasp_pose = pbu.multiply(pose.get_pose(), pbu.invert(grasp.pregrasp))
    gripper_path = list(
        pbu.interpolate_poses(grasp_pose, pregrasp_pose, pos_step_size=pos_step_size)
    )
    return gripper_path


def create_grasp_attachment(robot, side, grasp, **kwargs):
    arm_group, gripper_group, tool_name = robot.manipulators[side]
    return grasp.create_attachment(
        robot, link=pbu.link_from_name(robot, tool_name, **kwargs)
    )


def plan_workspace_motion(
    robot, side, tool_waypoints, attachment=None, obstacles=[], max_attempts=2, **kwargs
):
    assert tool_waypoints

    ik_info = robot.ik_info[side]
    arm_group, _, tool_name = robot.manipulators[side]
    tool_link = pbu.link_from_name(robot, tool_name, **kwargs)

    arm_joints = robot.joint_groups[side]
    parts = [robot] + ([] if attachment is None else [attachment.child])

    collision_fn = pbu.get_collision_fn(
        robot,
        arm_joints,
        obstacles=[],
        attachments=[],
        self_collisions=SELF_COLLISIONS,
        disabled_collisions=robot.disabled_collisions,
        disable_collisions=DISABLE_ALL_COLLISIONS,
        custom_limits=robot.custom_limits,
        **kwargs
    )

    for attempts in range(max_attempts):
        if attempts > 0:
            joints = pbu.get_movable_joints(robot)
            ranges = [pbu.get_joint_limits(robot, joint) for joint in joints]
            initialization_sample = [random.uniform(r[0], r[1]) for r in ranges]
            pbu.set_joint_positions(robot, arm_joints, initialization_sample)

        for arm_conf in p.calculateInverseKinematics(
            bodyUniqueId=robot,
            endEffectorLinkIndex=tool_link,
            targetPosition=tool_waypoints[0][0],
            targetOrientation=tool_waypoints[0][1],
            jointDamping=[0.1] * len(arm_joints),
            **kwargs
        ):
            if collision_fn(arm_conf):
                continue
            arm_waypoints = [arm_conf]

            for tool_pose in tool_waypoints[1:]:
                # TODO: joint weights
                arm_conf = p.calculateInverseKinematics(
                    bodyUniqueId=robot,
                    endEffectorLinkIndex=tool_link,
                    targetPosition=tool_pose[0],
                    targetOrientation=tool_pose[1],
                    jointDamping=[0.1] * len(arm_joints),
                    **kwargs
                )

                if arm_conf is None:
                    break

                if collision_fn(arm_conf):
                    continue

                arm_waypoints.append(arm_conf)

            else:
                pbu.set_joint_positions(robot, arm_joints, arm_waypoints[-1], **kwargs)
                if attachment is not None:
                    attachment.assign()
                if (
                    any(
                        pbu.pairwise_collisions(
                            part,
                            obstacles,
                            max_distance=(COLLISION_DISTANCE + EPSILON),
                            **kwargs
                        )
                        for part in parts
                    )
                    and not DISABLE_ALL_COLLISIONS
                ):
                    continue
                arm_path = pbu.interpolate_joint_waypoints(
                    robot, arm_joints, arm_waypoints, **kwargs
                )

                if any(collision_fn(q) for q in arm_path):
                    continue

                print(
                    "Found path with {} waypoints and {} configurations after {} attempts".format(
                        len(arm_waypoints), len(arm_path), attempts + 1
                    )
                )

                return arm_path
    return None


#######################################################


def workspace_collision(
    robot,
    arm,
    gripper_path,
    grasp=None,
    open_gripper=True,
    obstacles=[],
    max_distance=0.0,
    **kwargs
):
    if DISABLE_ALL_COLLISIONS:
        return False
    side = robot.side_from_arm(arm)
    _, gripper_group, tool_name = robot.manipulators[side]
    gripper = robot.get_component(gripper_group)

    if open_gripper:
        # TODO: make a separate method?
        closed_conf, open_conf = robot.get_group_limits(gripper_group)
        gripper_joints = robot.get_component_joints(gripper_group)
        pbu.set_joint_positions(gripper, gripper_joints, open_conf, **kwargs)

    parent_from_tool = robot.get_parent_from_tool(side)
    parts = [gripper]  # , obj]
    if grasp is not None:
        parts.append(grasp.body)
    for i, gripper_pose in enumerate(
        gripper_path
    ):  # TODO: be careful about the initial pose
        pbu.set_pose(
            gripper, pbu.multiply(gripper_pose, pbu.invert(parent_from_tool)), **kwargs
        )
        if grasp is not None:
            pbu.set_pose(grasp.body, pbu.multiply(gripper_pose, grasp.value), **kwargs)
        # attachment.assign()
        distance = (
            (COLLISION_DISTANCE + EPSILON)
            if (i == len(gripper_path) - 1)
            else max_distance
        )

        if any(
            pbu.pairwise_collisions(part, obstacles, max_distance=distance, **kwargs)
            for part in parts
        ):
            return True

    return False


def plan_prehensile(robot, arm, obj, pose, grasp, environment=[], **kwargs):
    pose.assign()
    gripper_path = compute_gripper_path(pose, grasp)  # grasp -> pregrasp
    gripper_waypoints = gripper_path[:1] + gripper_path[-1:]
    if workspace_collision(
        robot, arm, gripper_path, grasp=None, obstacles=[], **kwargs
    ):
        return None
    side = robot.side_from_arm(arm)
    create_grasp_attachment(robot, side, grasp, **kwargs)
    # print(get_closest_distance(robot, pr2.get_group_joints(arm), parent_link, tool_link, grasp_pose, obj))
    arm_path = plan_workspace_motion(
        robot, side, gripper_waypoints, attachment=None, obstacles=[], **kwargs
    )
    return arm_path


def sample_attachment_base_confs(robot, obj, pose, environment=[], **kwargs):
    robot_saver = pbu.BodySaver(robot, **kwargs)  # TODO: reset the rest conf
    obstacles = environment  # + [obj]

    base_generator = robot.base_sample_gen(pose)

    for base_conf in base_generator:
        robot_saver.restore()
        pose.assign()
        base_conf = GroupConf(robot, robot.base_group, positions=base_conf, **kwargs)
        base_conf.assign()
        if pbu.pairwise_collisions(robot, obstacles, max_distance=COLLISION_DISTANCE):
            continue
        yield base_conf


def sample_visibility_base_confs(robot, obj, pose, environment=[], **kwargs):
    robot_saver = pbu.BodySaver(robot, **kwargs)  # TODO: reset the rest conf
    obstacles = environment
    base_generator = robot.base_sample_gen(pose)
    for base_conf in base_generator:
        robot_saver.restore()
        pose.assign()
        base_conf = GroupConf(robot, robot.base_group, positions=base_conf, **kwargs)
        base_conf.assign()
        if pbu.pairwise_collisions(robot, obstacles, max_distance=COLLISION_DISTANCE):
            continue
        yield base_conf


def sample_prehensive_base_confs(
    robot, arm, obj, pose, grasp, environment=[], **kwargs
):
    robot_saver = pbu.BodySaver(robot, **kwargs)  # TODO: reset the rest conf
    obstacles = environment

    gripper_path = compute_gripper_path(pose, grasp)
    if workspace_collision(
        robot, arm, gripper_path, grasp=None, obstacles=obstacles, **kwargs
    ):  # grasp):
        return

    base_generator = pbu.uniform_pose_generator(
        robot, gripper_path[0], reachable_range=(0.5, 1.0)
    )
    for base_conf in base_generator:
        robot_saver.restore()
        pose.assign()
        base_conf = GroupConf(robot, robot.base_group, positions=base_conf, **kwargs)
        base_conf.assign()
        if pbu.pairwise_collisions(robot, obstacles, max_distance=COLLISION_DISTANCE):
            continue
        yield base_conf


def set_closed_positions(robot, arm):
    _, gripper_group, _ = robot.manipulators[arm]
    closed_conf, _ = robot.get_group_limits(gripper_group)
    robot.set_group_positions(gripper_group, closed_conf)
    return closed_conf


def set_open_positions(robot, arm):
    _, gripper_group, _ = robot.manipulators[arm]
    _, open_conf = robot.get_group_limits(gripper_group)
    robot.set_group_positions(gripper_group, open_conf)
    return open_conf
