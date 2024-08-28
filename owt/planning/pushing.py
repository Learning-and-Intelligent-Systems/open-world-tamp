import math
import random
from itertools import islice

import numpy as np

import owt.pb_utils as pbu
from owt.planning.primitives import (GroupTrajectory, RelativePose, Sequence,
                                     Switch)
from owt.planning.samplers import plan_workspace_motion
from owt.simulation.entities import WORLD_BODY, ParentBody

PUSH_FEATURES = [
    "block_width",
    "block_length",
    "block_height",
    "push_yaw",
    "push_distance",
]
TOOL_POSE = pbu.Pose(euler=pbu.Euler(pitch=np.pi / 2))  # +x out of gripper arm


def get_end_pose(initial_pose, goal_pos2d):
    initial_z = pbu.point_from_pose(initial_pose)[2]
    orientation = pbu.quat_from_pose(initial_pose)
    goal_x, goal_y = goal_pos2d
    end_pose = ([goal_x, goal_y, initial_z], orientation)
    return end_pose


def get_push_feature(robot, arm, block_body, initial_pose, goal_pos2d, environment=[]):
    block_reference = pbu.unit_pose()
    _, (block_w, block_l, block_h) = pbu.approximate_as_prism(
        block_body, body_pose=block_reference
    )
    goal_pose = get_end_pose(initial_pose, goal_pos2d)
    difference_initial = pbu.point_from_pose(
        pbu.multiply(pbu.invert(initial_pose), goal_pose)
    )

    feature = {
        "arm_name": arm,
        "block_width": block_w,
        "block_length": block_l,
        "block_height": block_h,
        "push_yaw": pbu.get_yaw(difference_initial),
        "push_distance": pbu.get_length(difference_initial),
    }
    return feature


PUSH_PARAMETER = {
    "gripper_z": 0.02,
    "gripper_tilt": np.pi / 8,
    "delta_push": 0.0,
    "delta_yaw": 0.0,
}


def sample_push_contact(robot, body, feature, parameter, environment=[], under=False):
    arm = feature["arm_name"]
    push_yaw = feature["push_yaw"]
    GRIPPER_LINKS = {
        "left": "l_gripper_palm_link",
        "right": "r_gripper_palm_link",
    }

    center, (width, _, height) = pbu.approximate_as_prism(
        body, body_pose=pbu.Pose(euler=pbu.Euler(yaw=push_yaw))
    )
    max_backoff = width + 0.1  # TODO: add gripper bounding box
    tool_link = pbu.link_from_name(robot, robot.manipulators[arm.split("_")[0]])
    tool_pose = pbu.get_link_pose(robot, tool_link)
    gripper_link = pbu.link_from_name(robot, GRIPPER_LINKS[arm.split("_")[0]])
    collision_links = pbu.get_link_subtree(robot, gripper_link)

    urdf_from_center = pbu.Pose(point=center)
    reverse_z = pbu.Pose(euler=pbu.Euler(pitch=math.pi))
    rotate_theta = pbu.Pose(euler=pbu.Euler(yaw=push_yaw))
    translate_z = pbu.Pose(
        point=pbu.Point(z=parameter["gripper_z"])
    )  # Relative to center
    tilt_gripper = pbu.Pose(euler=pbu.Euler(pitch=parameter["gripper_tilt"]))

    grasps = []
    for i in range(1 + under):
        flip_gripper = pbu.Pose(euler=pbu.Euler(yaw=i * math.pi))
        for x in np.arange(0, max_backoff, step=0.01):
            translate_x = pbu.Pose(point=pbu.Point(x=-x))
            grasp_pose = pbu.multiply(
                flip_gripper,
                tilt_gripper,
                translate_x,
                translate_z,
                rotate_theta,
                reverse_z,
                pbu.invert(urdf_from_center),
            )
            pbu.set_pose(body, pbu.multiply(tool_pose, TOOL_POSE, grasp_pose))
            if not pbu.link_pairs_collision(robot, collision_links, body):
                grasps.append(grasp_pose)
                break
    return grasps


def get_push_goal_gen_fn(robot, environment=[]):
    def gen_fn(body, pose1, region):
        start_point = pbu.point_from_pose(pose1)
        distance_range = (0.15, 0.2)
        while True:
            theta = random.uniform(-np.pi, np.pi)
            distance = random.uniform(*distance_range)
            end_point2d = np.array(start_point[:2]) + distance * pbu.unit_from_theta(
                theta
            )
            end_pose = (
                np.append(end_point2d, [start_point[2]]),
                pbu.quat_from_pose(pose1),
            )
            pbu.set_pose(body, end_pose)
            yield end_point2d,

    return gen_fn


def cartesian_path_unsupported(body, path, surface):
    for pose in path:
        pbu.set_pose(body, pose)
        if not pbu.is_center_stable(body, surface, above_epsilon=np.inf):
            return True
    return False


COLLISION_BUFFER = 0.0


def body_pair_collision(body1, body2, collision_buffer=COLLISION_BUFFER):
    if body1 == body2:
        return False
    return pbu.pairwise_collision(body1, body2, max_distance=collision_buffer)


def cartesian_path_collision(body, path, obstacles, **kwargs):
    for pose in path:
        pbu.set_pose(body, pose)
        if any(body_pair_collision(body, obst, **kwargs) for obst in obstacles):
            return True
    return False


def get_plan_push_fn(
    robot,
    environment=[],
    max_samples=1,
    max_attempts=5,
    collisions=True,
    parameter_fns={},
    repeat=False,
    **kwargs
):
    environment = list(environment)
    robot_saver = pbu.BodySaver(robot, **kwargs)
    side = robot.get_arbitrary_side()
    arm_group, gripper_group, tool_name = robot.manipulators[side]
    robot.get_component(gripper_group)

    # TODO(caelan): could also simulate the predicated sample
    # TODO(caelan): make final the orientation be aligned with gripper

    robot.disabled_collisions
    backoff_distance = 0.03
    approach_tform = pbu.Pose(point=np.array([-0.1, 0, 0]))  # Tool coordinates
    push_goal_gen_fn = get_push_goal_gen_fn(robot, environment)

    def gen_fn(arm, body, relative_pose1, region, region_pose, shape, base_conf):
        robot_saver.restore()
        base_conf.assign()

        pose1 = relative_pose1.get_pose()
        goals = push_goal_gen_fn(body, pose1, region)
        for (goal_pos2d,) in islice(goals, max_samples):
            if goal_pos2d is None:
                continue
            pose2 = get_end_pose(pose1, goal_pos2d)
            body_path = list(pbu.interpolate_poses(pose1, pose2))
            if cartesian_path_collision(
                body, body_path, set(environment) - {region}
            ) or cartesian_path_unsupported(body, body_path, region):
                continue
            push_direction = np.array(pbu.point_from_pose(pose2)) - np.array(
                pbu.point_from_pose(pose1)
            )
            backoff_tform = pbu.Pose(
                -backoff_distance * pbu.get_unit_vector(push_direction)
            )  # World coordinates
            feature = get_push_feature(
                robot, arm, body, pose1, goal_pos2d, environment=environment
            )
            push_contact = next(
                iter(
                    sample_push_contact(
                        robot,
                        body,
                        feature,
                        PUSH_PARAMETER,
                        environment=environment,
                        under=False,
                    )
                )
            )
            gripper_path = [
                pbu.multiply(pose, pbu.invert(pbu.multiply(TOOL_POSE, push_contact)))
                for pose in body_path
            ]

            push_path = plan_workspace_motion(
                robot, side, gripper_path, attachment=None, obstacles=environment
            )
            if push_path is None:
                continue

            pre_backoff_pose = pbu.multiply(backoff_tform, gripper_path[0])
            pre_approach_pose = pbu.multiply(pre_backoff_pose, approach_tform)
            pre_path = plan_workspace_motion(
                robot,
                side,
                [pre_backoff_pose, pre_approach_pose],
                attachment=None,
                obstacles=environment,
            )
            if pre_path is None:
                continue

            pre_path = pre_path[::-1]
            post_backoff_pose = pbu.multiply(backoff_tform, gripper_path[-1])
            post_approach_pose = pbu.multiply(post_backoff_pose, approach_tform)
            post_path = plan_workspace_motion(
                robot,
                side,
                [post_backoff_pose, post_approach_pose],
                attachment=None,
                obstacles=environment,
            )
            if post_path is None:
                continue

            robot.get_group_joints(arm_group)

            pre_arm_traj = GroupTrajectory(
                robot,
                arm_group,
                path=pre_path,
                contexts=[pose1],
                contact_links=robot.get_finger_links(
                    robot.get_group_joints(gripper_group)
                ),
                time_after_contact=1e-1,
            )
            arm_traj = GroupTrajectory(
                robot,
                arm_group,
                path=push_path,
                contexts=[pose1],
                contact_links=robot.get_finger_links(
                    robot.get_group_joints(gripper_group)
                ),
                time_after_contact=1e-1,
            )
            post_arm_traj = GroupTrajectory(
                robot,
                arm_group,
                path=post_path,
                contexts=[pose1],
                contact_links=robot.get_finger_links(
                    robot.get_group_joints(gripper_group)
                ),
                time_after_contact=1e-1,
            )

            switch = Switch(
                body,
                parent=ParentBody(
                    body=robot, link=pbu.link_from_name(robot, tool_name)
                ),
            )
            switch_off = Switch(body, parent=WORLD_BODY)
            commands = [
                pre_arm_traj,
                # switch,
                arm_traj,
                # switch_off,
                post_arm_traj,
            ]
            sequence = Sequence(commands=commands, name="push-{}-{}".format(arm, body))

            pbu.set_pose(body, pose2)
            push_pose = RelativePose(
                body, parent=ParentBody(region), parent_state=region_pose
            )
            return (push_pose, commands[0].first(), commands[-1].last(), sequence)

    return gen_fn
