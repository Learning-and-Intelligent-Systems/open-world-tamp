#!/usr/bin/env python3

from __future__ import print_function

import os
import warnings

import numpy as np
import pybullet_tools
from pybullet_tools.pr2_utils import (CLEAR_LEFT_ARM, LEFT_ARM, PR2_GROUPS,
                                      PR2_TOOL_FRAMES, RIGHT_ARM,
                                      arm_from_side, gripper_from_side,
                                      open_gripper, rightarm_from_leftarm,
                                      side_from_arm)
from pybullet_tools.utils import (PI, custom_limits_from_base_limits,
                                  link_from_name)

from owt.simulation.entities import Robot
from owt.simulation.lis import PR2_INFOS

pybullet_tools.utils.TEMP_DIR = "temp_meshes/"  # TODO: resolve conflict with pddlstream

from pybullet_tools.utils import PI, get_max_limits, link_from_name

from owt.simulation.entities import Camera, Manipulator
from owt.simulation.lis import (CAMERA_FRAME, CAMERA_MATRIX,
                                CAMERA_OPTICAL_FRAME, PR2_INFOS)

DEFAULT_LEFT_ARM = CLEAR_LEFT_ARM

from pybullet_tools.utils import link_from_name, user_input

from owt.simulation.controller import SimulatedController
# from run_estimator import create_parser
from owt.simulation.environment import set_gripper_friction
# TODO: all ROS should be the last import otherwise segfaults
from robots.pr2.pr2_controller import PR2Controller

PR2_PATH = os.path.abspath("models/ltamp/pr2_description/pr2.urdf")

warnings.filterwarnings("ignore")  # , category=DeprecationWarning)

from pybullet_tools.utils import Pose, add_data_path

from owt.simulation.environment import create_floor_object
from owt.simulation.lis import CAMERA_OPTICAL_FRAME

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


def get_input(message, options):
    full_message = "{} [{}]: ".format(message, ",".join(options))
    response = user_input(full_message)
    while response not in options:
        response = user_input(full_message)
    return response


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
        self.arms = [arm + "_arm" for arm in arms]

        self.real_execute = real_execute
        self.real_camera = real_camera

        base_side = 5.0
        base_limits = -base_side * np.ones(2) / 2, +base_side * np.ones(2) / 2
        custom_limits = custom_limits_from_base_limits(
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
                    link=link_from_name(robot_body, CAMERA_FRAME, client=client),
                    optical_frame=link_from_name(
                        robot_body, CAMERA_OPTICAL_FRAME, client=client
                    ),
                    camera_matrix=CAMERA_MATRIX,
                    client=client,
                )
            ]
        else:
            cameras = []

        manipulators = {
            side_from_arm(arm): Manipulator(
                arm_from_side(side_from_arm(arm)),
                gripper_from_side(side_from_arm(arm)),
                PR2_TOOL_FRAMES[side_from_arm(arm)],
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
            ik_info=PR2_INFOS,
            manipulators=manipulators,
            disabled_collisions=PR2_DISABLED_COLLISIONS,
            client=client,
            **kwargs
        )

    # def base_sample_gen(self, pose):
    #     return directed_pose_generator(
    #         self.robot, pose.get_pose(), reachable_range=(0.8, 0.8)
    #     )

    def get_default_conf(self, torso=0.25, tilt=PI / 3):
        conf = {
            "left_arm": DEFAULT_LEFT_ARM,
            "right_arm": rightarm_from_leftarm(DEFAULT_LEFT_ARM),
            "torso": [torso],
            "head": [0, tilt],
        }
        conf.update(
            {
                gripper: get_max_limits(
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
        open_gripper(self.robot, LEFT_ARM, client=self.client)
        open_gripper(self.robot, RIGHT_ARM, client=self.client)
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
