import os

# from run_estimator import *

from pybullet_tools.ikfast.utils import IKFastInfo
from pybullet_tools.utils import link_from_name
from open_world.simulation.entities import Camera, Manipulator, Robot
from open_world.simulation.lis import CAMERA_MATRIX as SIMULATED_CAMERA_MATRIX
from open_world.simulation.policy import Policy
from robots.panda.panda_controller import PandaController, SimulatedPandaController

CAMERA_FRAME = "camera_frame"
CAMERA_OPTICAL_FRAME = "camera_frame"
PANDA_INFO = IKFastInfo(
    module_name="panda.ikfast_panda_arm",
    base_link="panda_link0",
    ee_link="panda_link8",
    free_joints=["panda_joint7"],
)
PANDA_PATH = os.path.abspath("models/srl/franka_description/robots/panda_arm_hand.urdf")


class PandaRobot(Robot):
    def __init__(self, robot_body, link_names={}, client=None, *args, **kwargs):

        self.link_names = link_names
        self.body = robot_body
        self.client = client
        self.arms = ["main_arm"]

        PANDA_GROUPS = {
            "base": [],
            "main_arm": ["panda_joint{}".format(i) for i in range(1, 8)],
            "main_gripper": ["panda_finger_joint1", "panda_finger_joint2"],
        }

        PANDA_TOOL_FRAMES = {
            "main_arm": "panda_tool_tip",  # l_gripper_palm_link | l_gripper_tool_frame
        }

        panda_manipulators = {
            side_from_arm(arm): Manipulator(
                arm, gripper_from_arm(arm), PANDA_TOOL_FRAMES[arm]
            )
            for arm in self.arms
        }
        panda_ik_infos = {side_from_arm(arm): PANDA_INFO for arm in self.arms}

        if kwargs["args"].simulated:
            cameras = [
                Camera(
                    self,
                    link=link_from_name(self.body, CAMERA_FRAME, client=client),
                    optical_frame=link_from_name(
                        self.body, CAMERA_OPTICAL_FRAME, client=client
                    ),
                    camera_matrix=SIMULATED_CAMERA_MATRIX,
                    client=client,
                )
            ]
        else:
            cameras = []

        super(PandaRobot, self).__init__(
            robot_body,
            ik_info=panda_ik_infos,
            manipulators=panda_manipulators,
            cameras=cameras,
            joint_groups=PANDA_GROUPS,
            link_names=link_names,
            client=client,
            *args,
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

    def arm_from_side(self, side):
        return arm_from_side(side)

    def side_from_arm(self, arm):
        return side_from_arm(arm)

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
    def default_mobile_base_arm(self):
        return self.get_default_conf()["main_arm"]

    @property
    def default_fixed_base_arm(self):
        return self.get_default_conf()["main_arm"]

    @property
    def base_link(self):
        return link_from_name(self.robot, self.BASE_LINK, client=self.client)


class PandaPolicy(Policy):
    def __init__(self, args, robot, client=None, **kwargs):
        self.args = args
        self.robot = robot
        self.client = client
        super(PandaPolicy, self).__init__(args, robot, client=client, **kwargs)

    def reset_robot(self, **kwargs):
        conf = self.robot.get_default_conf()
        for group, positions in conf.items():
            if not self.args.simulated:
                group_dict = {
                    name: pos
                    for pos, name in zip(positions, self.robot.joint_groups[group])
                }
                self.controller.command_group_dict(group, group_dict)
            else:
                self.robot.set_group_positions(group, positions)

    def make_controller(self):
        if self.args.simulated:
            return SimulatedPandaController(self.robot, client=self.client)
        else:
            return PandaController(self.robot, client=self.client)


def side_from_arm(arm):
    side = arm.split("_")[0]
    return side


def arm_from_side(side):
    return "{}_arm".format(side)


def gripper_from_arm(arm):  # TODO: deprecate
    side = side_from_arm(arm)
    return "{}_gripper".format(side)
