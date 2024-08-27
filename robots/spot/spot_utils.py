import os

from open_world.simulation.controller import SimulatedController
from open_world.simulation.entities import Manipulator, Robot

from robots.spot.spot_controller import SpotController

SPOT_URDF = "models/srl/spot_description/mobile_model.urdf"
SPOT_PATH = os.path.abspath(SPOT_URDF)

SPOT_DISBLED_COLLISIONS = []
SPOT_GROUPS = {
    "base": ["x", "y", "theta"],
    "arm": [
        "arm0.sh0",
        "arm0.sh1",
        "arm0.hr0",
        "arm0.el0",
        "arm0.el1",
        "arm0.wr0",
        "arm0.wr1",
    ],
    "gripper": ["arm0.f1x"],
}

DEFAULT_JOINTS = {
    "x": 0,
    "y": 0,
    "theta": 0,
    "arm0.sh0": 0,
    "arm0.sh1": 0,
    "arm0.hr0": 0,
    "arm0.el0": 0,
    "arm0.el1": 0,
    "arm0.wr0": 0,
    "arm0.wr1": 0,
    "arm0.f1x": 0,
}


SPOT_IK = {}


class SpotRobot(Robot):
    def __init__(
        self, robot_body, client=None, real_execute=False, real_camera=False, **kwargs
    ):

        self.real_execute = real_execute
        self.real_camera = real_camera
        self.body = robot_body
        self.client = client

        movo_manipulators = {"arm": Manipulator("arm", "gripper", "gripper_tool_frame")}
        cameras = []

        if not self.real_execute:
            self.controller = SimulatedController(self.robot, client=self.client)
        else:
            self.controller = SpotController(self.args, self.robot, client=self.client)

        super(SpotRobot, self).__init__(
            robot_body,
            joint_groups=SPOT_GROUPS,
            manipulators=movo_manipulators,
            ik_info=SPOT_IK,
            cameras=cameras,
            disabled_collisions=SPOT_DISBLED_COLLISIONS,
            client=client,
            **kwargs
        )

    def get_default_conf(self):
        return {k: [DEFAULT_JOINTS[j] for j in v] for k, v in SPOT_GROUPS.items()}

    def reset(self, **kwargs):
        conf = self.get_default_conf(**kwargs)
        for group, positions in conf.items():
            if self.real_execute:
                raise NotImplementedError
            else:
                # If in sim, we can just set the joint positions
                self.set_group_positions(group, positions)
