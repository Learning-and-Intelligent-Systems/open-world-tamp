import os

import owt.pb_utils as pbu
from owt.simulation.controller import SimulatedController
from owt.simulation.entities import Camera, Manipulator, Robot
from owt.simulation.lis import CAMERA_MATRIX as SIMULATED_CAMERA_MATRIX
from robots.panda.panda_controller import PandaController

CAMERA_FRAME = "camera_frame"
CAMERA_OPTICAL_FRAME = "camera_frame"
PANDA_PATH = os.path.abspath("models/srl/franka_panda/panda.urdf")


class PandaRobot(Robot):
    def __init__(
        self,
        robot_body,
        link_names={},
        client=None,
        real_camera=False,
        real_execute=False,
        arms=["main_arm"],
        **kwargs
    ):
        self.link_names = link_names
        self.body = robot_body
        self.client = client
        self.arms = arms
        self.real_camera = real_camera
        self.real_execute = real_execute

        PANDA_GROUPS = {
            "base": [],
            "main_arm": ["panda_joint{}".format(i) for i in range(1, 8)],
            "main_gripper": ["panda_finger_joint1", "panda_finger_joint2"],
        }

        PANDA_TOOL_FRAMES = {
            "main_arm": "panda_tool_tip",
        }

        panda_manipulators = {
            arm: Manipulator(arm, arm.replace("arm", "gripper"), PANDA_TOOL_FRAMES[arm])
            for arm in self.arms
        }

        if not real_camera:
            cameras = [
                Camera(
                    self,
                    link=pbu.link_from_name(self.body, CAMERA_FRAME, client=client),
                    optical_frame=pbu.link_from_name(
                        self.body, CAMERA_OPTICAL_FRAME, client=client
                    ),
                    camera_matrix=SIMULATED_CAMERA_MATRIX,
                    client=client,
                )
            ]
        else:
            cameras = []

        if not self.real_execute:
            self.controller = SimulatedController(self.robot, client=self.client)
        else:
            self.controller = PandaController(self.robot, client=self.client)

        super(PandaRobot, self).__init__(
            robot_body,
            manipulators=panda_manipulators,
            cameras=cameras,
            joint_groups=PANDA_GROUPS,
            link_names=link_names,
            client=client,
            **kwargs
        )
        self.max_depth = 3.0
        self.min_z = 0.0
        self.BASE_LINK = "panda_link0"
        self.MAX_PANDA_FINGER = 0.045

    def get_default_conf(self):
        conf = {
            "main_arm": [
                -0.7102168405069942,
                -0.9392536020924664,
                1.9823867153201185,
                -1.4150627551046624,
                -0.1586580204963684,
                1.3177407226430045,
                2.7953193752934617,
            ],
            "main_gripper": [self.MAX_PANDA_FINGER, self.MAX_PANDA_FINGER],
        }
        return conf

    def arm_conf(self, arm, config):
        return config

    def get_closed_positions(self):
        return {"panda_finger_joint1": 0, "panda_finger_joint2": 0}

    def get_open_positions(self):
        return {
            "panda_finger_joint1": self.MAX_PANDA_FINGER,
            "panda_finger_joint2": self.MAX_PANDA_FINGER,
        }

    @property
    def groups(self):
        return self.joint_groups

    @property
    def default_fixed_base_arm(self):
        return self.get_default_conf()["main_arm"]

    @property
    def base_link(self):
        return pbu.link_from_name(self.robot, self.BASE_LINK, client=self.client)

    def reset(self, **kwargs):
        conf = self.get_default_conf()
        for group, positions in conf.items():
            if self.real_execute:
                group_dict = {
                    name: pos for pos, name in zip(positions, self.joint_groups[group])
                }
                self.controller.command_group_dict(group, group_dict)
            else:
                self.set_group_positions(group, positions)
