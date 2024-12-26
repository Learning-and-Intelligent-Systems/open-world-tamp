#!/usr/bin/env python3

from __future__ import print_function

import os
import warnings

import numpy as np

import owt.pb_utils as pbu
from owt.simulation.entities import Camera, Manipulator, Robot
from owt.simulation.lis import (CAMERA_FRAME, CAMERA_MATRIX,
                                CAMERA_OPTICAL_FRAME)

LEFT_ARM = "left"
RIGHT_ARM = "right"
CLEAR_LEFT_ARM = [np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0]
DEFAULT_LEFT_ARM = CLEAR_LEFT_ARM

from owt.simulation.controller import SimulatedController
from owt.simulation.environment import set_gripper_friction
from robots.pr2.pr2_controller import PR2Controller

PR2_PATH = os.path.abspath("models/ltamp/pr2_description/pr2.urdf")

warnings.filterwarnings("ignore")

from owt.simulation.lis import CAMERA_OPTICAL_FRAME

ARM_NAMES = (LEFT_ARM, RIGHT_ARM)


def side_from_arm(arm):
    side = arm.split("_")[0]
    assert side in ARM_NAMES
    return side


side_from_gripper = side_from_arm


def arm_from_arm(arm):  # TODO: deprecate
    side = side_from_arm(arm)
    assert side in ARM_NAMES
    return "{}_arm".format(side)


arm_from_side = arm_from_arm


def gripper_from_arm(arm):  # TODO: deprecate
    side = side_from_arm(arm)
    assert side in ARM_NAMES
    return "{}_gripper".format(side)


gripper_from_side = gripper_from_arm

PR2_GROUPS = {
    "base": ["x", "y", "theta"],
    "torso": ["torso_lift_joint"],
    "head": ["head_pan_joint", "head_tilt_joint"],
    arm_from_arm(LEFT_ARM): [
        "l_shoulder_pan_joint",
        "l_shoulder_lift_joint",
        "l_upper_arm_roll_joint",
        "l_elbow_flex_joint",
        "l_forearm_roll_joint",
        "l_wrist_flex_joint",
        "l_wrist_roll_joint",
    ],
    arm_from_arm(RIGHT_ARM): [
        "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint",
        "r_elbow_flex_joint",
        "r_forearm_roll_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint",
    ],
    gripper_from_arm(LEFT_ARM): [
        "l_gripper_l_finger_joint",
        "l_gripper_r_finger_joint",
        "l_gripper_l_finger_tip_joint",
        "l_gripper_r_finger_tip_joint",
    ],
    gripper_from_arm(RIGHT_ARM): [
        "r_gripper_l_finger_joint",
        "r_gripper_r_finger_joint",
        "r_gripper_l_finger_tip_joint",
        "r_gripper_r_finger_tip_joint",
    ],
    # r_gripper_joint & l_gripper_joint are not mimicked
}


HEAD_LINK_NAME = "high_def_optical_frame"  # high_def_optical_frame | high_def_frame | wide_stereo_l_stereo_camera_frame

PR2_TOOL_FRAMES = {
    LEFT_ARM: "l_gripper_tool_frame",  # l_gripper_palm_link | l_gripper_tool_frame
    RIGHT_ARM: "r_gripper_tool_frame",  # r_gripper_palm_link | r_gripper_tool_frame
    "head": HEAD_LINK_NAME,
}


def rightarm_from_leftarm(config):
    right_from_left = np.array([-1, 1, -1, 1, -1, 1, -1])
    return config * right_from_left


PR2_DISABLED_COLLISIONS = [
    (76, 3),
    (77, 18),
    (76, 78),
    (76, 79),
    (79, 18),
    (77, 79),
    (79, 81),
    (82, 84),
    (82, 85),
    (82, 87),
    (83, 85),
    (83, 87),
    (79, 82),
    (82, 93),
    (52, 3),
    (53, 18),
    (52, 54),
    (52, 55),
    (53, 55),
    (55, 57),
    (58, 60),
    (58, 65),
    (52, 3),
    (82, 91),
    (55, 58),
    (55, 18),
]


class PR2Robot(Robot):
    def __init__(
        self,
        robot_body,
        client=None,
        real_camera=False,
        real_execute=False,
        arms=[LEFT_ARM],
        **kwargs
    ):
        self.arms = arms

        self.real_execute = real_execute
        self.real_camera = real_camera

        base_side = 5.0
        base_limits = -base_side * np.ones(2) / 2, +base_side * np.ones(2) / 2
        custom_limits = pbu.custom_limits_from_base_limits(
            robot_body, base_limits, client=client
        )

        self.joint_groups = PR2_GROUPS
        self.body = robot_body
        self.max_depth = float("inf")
        self.client = client
        self.min_z = 0.0

        self.set_default_conf()
        set_gripper_friction(self, client=self.client)

        if not real_camera:
            cameras = [
                Camera(
                    self,
                    link=pbu.link_from_name(robot_body, CAMERA_FRAME, client=client),
                    optical_frame=pbu.link_from_name(
                        robot_body, CAMERA_OPTICAL_FRAME, client=client
                    ),
                    camera_matrix=CAMERA_MATRIX,
                    client=client,
                )
            ]
        else:
            cameras = []

        manipulators = {
            arm: Manipulator(
                arm,
                arm.replace("arm", "gripper"),
                PR2_TOOL_FRAMES[arm],
            )
            for arm in self.arms
        }

        if not real_execute:
            self.controller = SimulatedController(self, client=self.client)
        else:
            self.controller = PR2Controller(self, client=self.client)

        super(PR2Robot, self).__init__(
            robot_body,
            joint_groups=PR2_GROUPS,
            custom_limits=custom_limits,
            cameras=cameras,
            manipulators=manipulators,
            disabled_collisions=PR2_DISABLED_COLLISIONS,
            client=client,
            **kwargs
        )

    def get_gripper_joints(self, arm):
        return self.get_group_joints(gripper_from_arm(arm))

    def set_gripper_position(self, arm, position):
        gripper_joints = self.get_gripper_joints(arm)
        pbu.set_joint_positions(
            self.robot, gripper_joints, [position] * len(gripper_joints)
        )

    def open_arm(self, arm):  # These are mirrored on the pr2
        for joint in self.get_gripper_joints(arm):
            pbu.set_joint_position(
                self.robot,
                joint,
                pbu.get_max_limit(self.robot, joint, client=self.client),
                client=self.client,
            )

    def close_arm(self, arm):
        for joint in self.get_gripper_joints(arm):
            pbu.set_joint_position(
                self.robot,
                joint,
                pbu.get_min_limit(self.robot, joint, client=self.client),
                client=self.client,
            )

    def get_default_conf(self, torso=0.25, tilt=np.pi / 3):
        conf = {
            "left_arm": DEFAULT_LEFT_ARM,
            "right_arm": rightarm_from_leftarm(DEFAULT_LEFT_ARM),
            "torso": [torso],
            "head": [0, tilt],
        }
        conf.update(
            {
                gripper: pbu.get_max_limits(
                    self.robot, self.get_group_joints(gripper), client=self.client
                )
                for gripper in self.gripper_groups
            }
        )

        return conf

    def set_default_conf(self):
        group_positions = self.get_default_conf()
        group_positions.update(
            {
                "base": [-0.5, 0, 0],
                "torso": [0.2],  # TODO: varying torso limits
            }
        )

        for group, positions in group_positions.items():
            self.set_group_positions(group, positions)
        self.open_arm(LEFT_ARM)
        self.open_arm(RIGHT_ARM)
        return group_positions

    def reset(self, **kwargs):
        conf = self.get_default_conf(**kwargs)
        clients = []
        for group in conf:
            if group in [self.base_group]:  # + robot.gripper_groups:
                continue
            positions = conf[group]
            if self.real_execute:
                client = self.controller.command_group(
                    group, positions, timeout=5.0, blocking=False
                )
                clients.append(client)
            else:
                self.set_group_positions(group, positions)
        return self.update_conf()
