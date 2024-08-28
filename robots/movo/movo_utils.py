import copy
import os

import numpy as np

import owt.pb_utils as pbu
from owt.planning.primitives import GroupConf
from owt.planning.streams import get_plan_motion_fn
from owt.simulation.controller import SimulatedController
from owt.simulation.entities import Camera, Manipulator, Robot
from owt.simulation.lis import CAMERA_MATRIX as SIMULATED_CAMERA_MATRIX
from robots.movo.movo_controller import MovoController
from robots.movo.movo_sender import (get_color_image, get_depth_image,
                                     get_pointcloud)

MOVO_URDF = "models/srl/movo_description/movo_robotiq_collision.urdf"
MOVO_PATH = os.path.abspath(MOVO_URDF)

LEFT = "left"
RIGHT = "right"

ARMS = ["{}_arm".format(RIGHT), "{}_arm".format(LEFT)]
SIDE = [RIGHT, LEFT]

BASE_JOINTS = ["x", "y", "theta"]
TORSO_JOINTS = ["linear_joint"]
HEAD_JOINTS = ["pan_joint", "tilt_joint"]

ARM_JOINTS = [
    "{}_shoulder_pan_joint",
    "{}_shoulder_lift_joint",
    "{}_arm_half_joint",
    "{}_elbow_joint",
    "{}_wrist_spherical_1_joint",
    "{}_wrist_spherical_2_joint",
    "{}_wrist_3_joint",
]

KG3_GRIPPER_JOINTS = [
    "{}_gripper_finger1_joint",
    "{}_gripper_finger2_joint",
    "{}_gripper_finger3_joint",
]

ROBOTIQ_GRIPPER_JOINTS = [
    "{}_gripper_finger1_joint",
    "{}_gripper_finger2_joint",
    "{}_gripper_finger1_inner_knuckle_joint",
    "{}_gripper_finger1_finger_tip_joint",
    "{}_gripper_finger2_inner_knuckle_joint",
    "{}_gripper_finger2_finger_tip_joint",
]

EE_LINK = "{}_ee_link"
TOOL_LINK = "{}_tool_link"


COMMAND_MOVO_GROUPS = {
    "base": ["x", "y", "theta", "linear_joint"],
    "left_arm": [
        "left_shoulder_pan_joint",
        "left_shoulder_lift_joint",
        "left_arm_half_joint",
        "left_elbow_joint",
        "left_wrist_spherical_1_joint",
        "left_wrist_spherical_2_joint",
        "left_wrist_3_joint",
    ],
    "right_arm": [
        "right_shoulder_pan_joint",
        "right_shoulder_lift_joint",
        "right_arm_half_joint",
        "right_elbow_joint",
        "right_wrist_spherical_1_joint",
        "right_wrist_spherical_2_joint",
        "right_wrist_3_joint",
    ],
    "left_gripper": [
        "left_gripper_finger1_joint",
        "left_gripper_finger2_joint",
        "left_gripper_finger3_joint",
    ],
    "right_gripper": ["right_gripper_finger1_joint"],
    "head": ["pan_joint", "tilt_joint"],
}

MOVO_GROUPS = {
    "base": ["x", "y", "theta", "linear_joint"],
    "left_arm": [
        "left_shoulder_pan_joint",
        "left_shoulder_lift_joint",
        "left_arm_half_joint",
        "left_elbow_joint",
        "left_wrist_spherical_1_joint",
        "left_wrist_spherical_2_joint",
        "left_wrist_3_joint",
    ],
    "right_arm": [
        "right_shoulder_pan_joint",
        "right_shoulder_lift_joint",
        "right_arm_half_joint",
        "right_elbow_joint",
        "right_wrist_spherical_1_joint",
        "right_wrist_spherical_2_joint",
        "right_wrist_3_joint",
    ],
    "left_gripper": [
        "left_gripper_finger1_joint",
        "left_gripper_finger2_joint",
        "left_gripper_finger3_joint",
        "left_gripper_finger1_finger_tip_joint",
        "left_gripper_finger2_finger_tip_joint",
        "left_gripper_finger3_finger_tip_joint",
    ],
    "right_gripper": [
        "right_gripper_finger1_inner_knuckle_joint",
        "right_gripper_finger1_joint",
        "right_gripper_finger1_finger_tip_joint",
        "right_gripper_finger2_inner_knuckle_joint",
        "right_gripper_finger2_joint",
        "right_gripper_finger2_finger_tip_joint",
    ],
    "head": ["pan_joint", "tilt_joint"],
}

MOVO_TOOL_FRAMES = {"left_arm": "left_tool_link", "right_arm": "right_tool_link"}

MOVO_DISABLED_COLLISIONS = [
    (36, 4),
    (36, 5),
    (37, 5),
    (41, 45),
    (16, 4),
    (16, 5),
    (21, 25),
    (21, 24),
]

KINECT_INTRINSICS = [
    528.6116160556213,
    0.0,
    477.68448358339145,
    0.0,
    531.8537610715608,
    255.95470886737945,
    0.0,
    0.0,
    1.0,
]
BASE_LINK = "base_link"

MOVO_CLOSED_CONF = {
    "right_gripper_finger1_joint": 0.7929515,
    "right_gripper_finger2_joint": 0.7929515,
    "right_gripper_finger1_inner_knuckle_joint": 0.7929515,
    "right_gripper_finger2_inner_knuckle_joint": 0.7929515,
    "right_gripper_finger1_finger_tip_joint": -0.7929515,
    "right_gripper_finger2_finger_tip_joint": -0.7929515,
}

MOVO_OPEN_CONF = {
    "right_gripper_finger1_joint": 0,
    "right_gripper_finger2_joint": 0,
    "right_gripper_finger1_inner_knuckle_joint": 0,
    "right_gripper_finger2_inner_knuckle_joint": 0,
    "right_gripper_finger1_finger_tip_joint": 0,
    "right_gripper_finger2_finger_tip_joint": 0,
}

# Arms down
DEFAULT_JOINTS = {
    "pan_joint": -0.07204942405223846,
    "tilt_joint": -0.599216890335083,
    "left_shoulder_pan_joint": 1.0,
    "left_shoulder_lift_joint": 1.9619225455538198,
    "left_arm_half_joint": 0.13184053877842938,
    "left_elbow_joint": 1.8168894557491948,
    "left_wrist_spherical_1_joint": -0.30988063075165684,
    "left_wrist_spherical_2_joint": -1.753361745316172,
    "left_wrist_3_joint": 1.725726522158583,
    "right_shoulder_pan_joint": -1,
    "right_shoulder_lift_joint": -1.9861489225161073,
    "right_arm_half_joint": 0.02609983172656305,
    "right_elbow_joint": -1.8699706504902727,
    "right_wrist_spherical_1_joint": 0.2607507015409034,
    "right_wrist_spherical_2_joint": 1.5755063934988107,
    "right_wrist_3_joint": -1.4726268826923956,
    "left_gripper_finger1_joint": -0.0008499202079690222,
    "left_gripper_finger2_joint": -0.0,
    "left_gripper_finger3_joint": 0.0,
    "right_gripper_finger1_joint": 0.0,
    "linear_joint": 0.3,
    "right_gripper_finger1_joint": 0,
    "right_gripper_finger2_joint": 0,
    "right_gripper_finger1_inner_knuckle_joint": 0,
    "right_gripper_finger2_inner_knuckle_joint": 0,
    "right_gripper_finger1_finger_tip_joint": 0,
    "right_gripper_finger2_finger_tip_joint": 0,
}


RIGHT_ATTACH_CONF = [
    -0.021811289327748895,
    -0.5591495793058756,
    0.09515283160149757,
    -0.9770537496674913,
    0.22921576166484137,
    1.059975131790689,
    -1.6935222466767996,
]
LEFT_ATTACH_CONF = [
    -0.2760957691629127,
    0.5009078441624968,
    0.2956304885223213,
    1.2349056669408707,
    -0.012336294801464476,
    -0.3835782875974208,
    1.7257314490066005,
]


class MovoRobot(Robot):
    def __init__(
        self,
        robot_body,
        client=None,
        real_execute=False,
        real_camera=False,
        arms=["right_arm"],
        **kwargs
    ):
        self.real_execute = real_execute
        self.real_camera = real_camera
        self.body = robot_body
        self.arms = arms
        self.client = client

        self.CAMERA_OPTICAL_FRAME = "kinect2_rgb_optical_frame"
        self.CAMERA_FRAME = "kinect2_rgb_link"
        movo_manipulators = {
            arm: Manipulator(arm, arm.replace("arm", "gripper"), MOVO_TOOL_FRAMES[arm])
            for arm in self.arms
        }

        if not real_camera:
            cameras = [
                Camera(
                    self,
                    link=pbu.link_from_name(
                        robot_body, self.CAMERA_OPTICAL_FRAME, client=self.client
                    ),
                    optical_frame=pbu.link_from_name(
                        robot_body, self.CAMERA_OPTICAL_FRAME, client=self.client
                    ),
                    camera_matrix=SIMULATED_CAMERA_MATRIX,
                    client=self.client,
                )
            ]
        else:
            cameras = []

        self.command_joint_groups = COMMAND_MOVO_GROUPS

        if not self.real_execute:
            self.controller = SimulatedController(self.robot, client=self.client)
        else:
            self.controller = MovoController(self.args, self.robot, client=self.client)

        super(MovoRobot, self).__init__(
            robot_body,
            joint_groups=MOVO_GROUPS,
            manipulators=movo_manipulators,
            cameras=cameras,
            disabled_collisions=MOVO_DISABLED_COLLISIONS,
            client=client,
            **kwargs
        )

        self.intrinsics = np.asarray(KINECT_INTRINSICS).reshape(3, 3)

    def directed_pose_generator(self, gripper_pose, **kwargs):
        point = pbu.point_from_pose(gripper_pose)
        while True:
            base_values = sample_directed_reachable_base(self, point, **kwargs)
            if base_values is None:
                break
            yield tuple(list(base_values) + [0.1])  # Append torso values

    def base_sample_gen(self, pose):
        return self.directed_pose_generator(pose.get_pose(), reachable_range=(0.7, 0.7))

    @property
    def base_group(self):
        return "base"

    def read_images(self):
        rgb_data = get_color_image()
        depth_data = get_depth_image()
        rgb_image = np.frombuffer(rgb_data["data"], dtype=np.uint8).reshape(
            rgb_data["height"], rgb_data["width"], -1
        )
        # NOTE original encoding=16UC1 (uint16, 1channel), in mm.
        depth_image = (
            np.frombuffer(depth_data["data"], dtype=np.uint16)
            .reshape(depth_data["height"], depth_data["width"])
            .astype(np.float32)
            / 1000
        )
        return rgb_image, depth_image

    @property
    def head_group(self):
        return "head"

    def get_default_conf(self):
        default_default_joints = copy.deepcopy(DEFAULT_JOINTS)
        get_jval = lambda j: (
            default_default_joints[j]
            if j in default_default_joints.keys()
            else pbu.get_joint_position(
                self,
                pbu.joint_from_name(self, j, client=self.client),
                client=self.client,
            )
        )
        return {k: [get_jval(j) for j in v] for k, v in MOVO_GROUPS.items()}

    def get_camera_pose(self):
        kinect2_link = pbu.link_from_name(self, "kinect2_rgb_optical_frame")
        kinect2_pose = pbu.get_link_pose(self, kinect2_link)
        return kinect2_pose

    def arm_conf(self, arm, config):
        return config

    def get_arbitrary_side(self):
        return list(self.manipulators.keys())[0]

    def get_open_positions(self):
        return

    def get_closed_positions(self):
        return

    def get_max_gripper_width(self, gripper_joints, **kwargs):
        with pbu.ConfSaver(self, client=self.client):
            pbu.set_joint_positions(
                self,
                gripper_joints,
                [
                    MOVO_OPEN_CONF[j]
                    for j in pbu.get_joint_names(
                        self, gripper_joints, client=self.client
                    )
                ],
                client=self.client,
            )
            return super().get_gripper_width(gripper_joints, **kwargs)

    @property
    def default_mobile_base_arm(self):
        return self.get_default_conf()["right_arm"]

    @property
    def default_fixed_base_arm(self):
        return self.get_default_conf()["right_arm"]

    def get_finger_links(self, gripper_joints):
        moving_links = pbu.get_moving_links(self, gripper_joints, client=self.client)
        shape_links = [
            link
            for link in moving_links
            if pbu.get_collision_data(self, link, client=self.client)
        ]
        link_names = pbu.get_link_names(self, shape_links, client=self.client)

        finger_links = [
            link
            for (linki, link) in enumerate(shape_links)
            if not any(
                pbu.get_collision_data(self, child, client=self.client)
                for child in pbu.get_link_children(self, link, client=self.client)
            )
            and "tip" in link_names[linki]
        ]
        if len(finger_links) != 2:
            raise RuntimeError(finger_links)
        return finger_links

    def get_group_limits(self, group):
        if group == "right_gripper":
            group_joints = self.get_group_joints(group)
            group_joint_min = [
                MOVO_CLOSED_CONF[j]
                for j in pbu.get_joint_names(self, group_joints, client=self.client)
            ]
            group_joint_max = [
                MOVO_OPEN_CONF[j]
                for j in pbu.get_joint_names(self, group_joints, client=self.client)
            ]
            return group_joint_min, group_joint_max
        else:
            return super().get_group_limits(group)

    def read_pointcloud(self):
        pointcloud_data = get_pointcloud()
        return pointcloud_data

    def get_parent_from_tool(self, manipulator):
        tool_link = self.get_tool_link(manipulator)
        parent_link = pbu.link_from_name(self, "right_wrist_3_link", client=self.client)
        return pbu.get_relative_pose(
            self.robot, tool_link, parent_link, client=self.client
        )

    @property
    def base_link(self):
        return pbu.link_from_name(self.robot, BASE_LINK, client=self.client)

    def reset(self, **kwargs):
        conf = self.get_default_conf(**kwargs)
        for group, positions in conf.items():
            if self.real_execute:
                # We need to go to default joints so we don't collide with things
                new = [pos for pos, name in zip(positions, MOVO_GROUPS[group])]
                if "arm" in group:
                    motion_gen = get_plan_motion_fn(self)
                    current = [
                        pbu.get_joint_position(self, pbu.joint_from_name(self, j))
                        for j in MOVO_GROUPS[group]
                    ]
                    q1 = GroupConf(self, group, positions=current)
                    q2 = GroupConf(self, group, positions=new)
                    (traj,) = motion_gen(group, q1, q2)
                    path = traj.commands[0].path
                    self.controller.command_group_trajectory(group, path, [], dt=0)
                else:
                    self.controller.command_group_dict(
                        group,
                        {name: pos for pos, name in zip(positions, MOVO_GROUPS[group])},
                    )
            else:
                # If in sim, we can just set the joint positions
                self.set_group_positions(group, positions)
