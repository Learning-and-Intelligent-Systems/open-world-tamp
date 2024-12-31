import time

from owt.simulation.controller import Controller
from robots.panda.panda_sender import (capture_realsense, close_gripper,
                                       command_arm, execute_position_path,
                                       get_joint_states, open_gripper)


class PandaController(Controller):
    def __init__(self, robot, verbose=True, **kwargs):
        self.robot = robot
        super(PandaController, self).__init__(**kwargs)

        self.arms = ["main_arm"]
        self.is_closed = False

    def command_group_trajectory(
        self, group, positions, times_from_start, blocking=True, **kwargs
    ):
        if group.endswith("_arm"):  # TODO: head
            pdicts = [
                {name: val for name, val in zip(self.robot.groups[group], position)}
                for position in positions
            ]
            execute_position_path(pdicts)
        elif group.endswith("_gripper"):
            for position in [positions[-1]]:
                pdict = {
                    name: val for name, val in zip(self.robot.groups[group], position)
                }
                self.command_gripper(pdict)
        else:
            raise NotImplementedError

    def any_arm_fully_closed(self):
        # return self.is_closed
        return False

    @property
    def joint_names(self):
        return list(self.joint_positions.keys())

    @property
    def joint_positions(self):
        arm_joints = get_joint_states()
        if self.is_closed:
            arm_joints.update(get_closed_positions())
        elif self.is_open:
            arm_joints.update(get_open_positions())
        return arm_joints

    @property
    def joint_velocities(self):
        return dict(zip(self.joint_state.name, self.joint_state.velocity))

    def wait_for_clients(self, clients, timeout=0):
        pass

    def command_arm(self, positions):
        command_arm(positions)

    def open_gripper(self, arm, blocking=True):
        open_gripper()

    def close_gripper(self, arm, blocking=True):
        close_gripper()

    def command_gripper(self, positions):
        if list(positions.values())[0] < 0.02:
            self.close_gripper(None)
            self.is_closed = True
        else:
            self.open_gripper(None)
            self.is_closed = False

    @property
    def is_open(self):
        return not self.is_closed

    def command_group_dict(self, group, positions, **kwargs):  # TODO: default timeout
        if group == "main_arm":
            return self.command_arm(positions, **kwargs)
        elif group == "main_gripper":
            return self.command_gripper(positions, **kwargs)
        else:
            raise NotImplementedError

    def capture_image(self):
        rgb, depth, intrinsics = capture_realsense()
        return rgb, depth, intrinsics, {}
