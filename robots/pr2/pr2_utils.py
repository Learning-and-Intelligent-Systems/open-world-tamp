#!/usr/bin/env python3

from __future__ import print_function

import sys
import warnings

import numpy as np

sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
    ]
)

import os

import numpy as np
from pybullet_tools.pr2_utils import (
    CLEAR_LEFT_ARM,
    LEFT_ARM,
    PR2_GROUPS,
    PR2_TOOL_FRAMES,
    RIGHT_ARM,
    arm_from_side,
    gripper_from_side,
    open_gripper,
    rightarm_from_leftarm,
    side_from_arm,
)
from pybullet_tools.utils import (
    PI,
    HideOutput,
    Pose,
    add_data_path,
    connect,
    custom_limits_from_base_limits,
    draw_pose,
    link_from_name,
    set_camera_pose,
    set_dynamics,
)

from open_world.simulation.entities import Robot
from open_world.simulation.environment import create_table_object
from open_world.simulation.lis import PR2_INFOS

import pybullet_tools
pybullet_tools.utils.TEMP_DIR = "temp_meshes/"  # TODO: resolve conflict with pddlstream

from pybullet_tools.utils import (
    PI,
    Pose,
    connect,
    draw_pose,
    get_max_limits,
    link_from_name,
)

from open_world.simulation.entities import Camera, Manipulator
from open_world.simulation.lis import (
    CAMERA_FRAME,
    CAMERA_MATRIX,
    CAMERA_OPTICAL_FRAME,
    PR2_INFOS,
)
from open_world.simulation.utils import get_rigid_ancestor

DEFAULT_LEFT_ARM = CLEAR_LEFT_ARM

import time

from pybullet_tools.utils import (
    link_from_name,
    user_input,
)

from open_world.estimation.geometry import cloud_from_depth, estimate_surface_mesh

# from run_estimator import create_parser
from open_world.estimation.tables import estimate_surfaces
from open_world.simulation.environment import set_gripper_friction
from open_world.simulation.policy import Policy

# TODO: all ROS should be the last import otherwise segfaults
from robots.pr2.pr2_controller import PR2Controller, SimulatedPR2Controller

PR2_PATH = os.path.abspath("models/ltamp/pr2_description/pr2.urdf")

warnings.filterwarnings("ignore")  # , category=DeprecationWarning)


from pybullet_tools.utils import Pose, add_data_path, connect

if __name__ == "__main__":
    connect(
        use_gui=True, width=None, height=None, shadows=False
    )  # TODO: Failed to retrieve a framebuffer config


from open_world.simulation.environment import create_floor_object
from open_world.simulation.lis import CAMERA_OPTICAL_FRAME

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

def review_plan(task, ros_world, plan, time_step=0.04):
    raise NotImplementedError()

class PR2Policy(Policy):
    def __init__(self, args, robot, client=None, **kwargs):
        self.args = args
        self.robot = robot
        self.client = client
        super(PR2Policy, self).__init__(args, robot, client=client, **kwargs)

    def make_controller(self):
        if not self.args.real:
            return SimulatedPR2Controller(self.robot, client=self.client)
        else:
            return PR2Controller(self.robot, client=self.client)

    def estimate_surfaces(self, camera_image, task):
        surfaces = estimate_surfaces(self.belief, camera_image, client=self.client)
        self.update_rendered_image()
        return surfaces

    def reset_robot(self, **kwargs):
        conf = self.robot.get_default_conf(**kwargs)
        clients = []
        for group in conf:
            if group in [self.robot.base_group]:  # + robot.gripper_groups:
                continue
            positions = conf[group]
            if self.args.real:
                client = self.controller.command_group(
                    group, positions, timeout=5.0, blocking=False
                )
                clients.append(client)
            else:
                self.robot.set_group_positions(group, positions)
        return self.update_robot()


class PR2Robot(Robot):
    def __init__(self, robot_body, client=None, *args, **kwargs):
        self.arms = ["left_arm"]
        base_side = 5.0
        base_limits = -base_side * np.ones(2) / 2, +base_side * np.ones(2) / 2
        custom_limits = custom_limits_from_base_limits(
            robot_body, base_limits, client=client
        )

        self.joint_groups = PR2_GROUPS
        self.body = robot_body
        self.max_depth = float("inf")
        self.client = client

        self.set_default_conf()
        set_gripper_friction(self, client=self.client)

        if not kwargs["args"].real:
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

        ik_info = PR2_INFOS  # TODO: use here within manipulator

        super(PR2Robot, self).__init__(
            robot_body,
            joint_groups=PR2_GROUPS,
            custom_limits=custom_limits,
            cameras=cameras,
            ik_info=ik_info,
            manipulators=manipulators,
            disabled_collisions=PR2_DISABLED_COLLISIONS,
            client=client,
            *args,
            **kwargs
        )

    def base_sample_gen(self, pose):
        return directed_pose_generator(
            self.robot, pose.get_pose(), reachable_range=(0.8, 0.8)
        )

        
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
        # dump_body(pr2, links=False)
        # torso_joints = robot.get_group_joints('torso')
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


def create_default_env(**kwargs):
    # TODO: p.loadSoftBody
    set_camera_pose(
        camera_point=[0.75, -0.75, 1.25], target_point=[-0.75, 0.75, 0.0], **kwargs
    )
    draw_pose(Pose(), length=1, **kwargs)

    add_data_path()
    with HideOutput(enable=True):
        create_floor_object(**kwargs)
        table = create_table_object(**kwargs)
        obstacles = [
            # floor, # collides with the robot when MAX_DISTANCE >= 5e-3
            table,
        ]

        for obst in obstacles:
            # print(get_dynamics_info(obst))
            set_dynamics(
                obst,
                lateralFriction=1.0,  # linear (lateral) friction
                spinningFriction=1.0,  # torsional friction around the contact normal
                rollingFriction=0.01,  # torsional friction orthogonal to contact normal
                restitution=0.0,  # restitution: 0 => inelastic collision, 1 => elastic collision
                **kwargs
            )

    return table, obstacles
