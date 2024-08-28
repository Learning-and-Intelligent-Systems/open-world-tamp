import time

import owt.pb_utils as pbu


class Controller(object):
    def __init__(self, *args, **kwargs):
        pass


class SimulatedController(Controller):
    def __init__(self, robot, client=None, **kwargs):
        self.client = client
        self.robot = robot

    def side_from_arm(self, arm):
        return arm.replace("_arm", "")

    def open_gripper(self, arm):  # These are mirrored on the pr2
        _, gripper_group, _ = self.robot.manipulators[self.side_from_arm(arm)]
        _, open_conf = self.robot.get_group_limits(gripper_group)
        self.command_group(gripper_group, open_conf)

    def close_gripper(self, arm):  # These are mirrored on the pr2
        _, gripper_group, _ = self.robot.manipulators[self.side_from_arm(arm)]
        closed_conf, _ = self.robot.get_group_limits(gripper_group)
        self.command_group(gripper_group, closed_conf)

    def get_group_joints(self, group):
        return pbu.joints_from_names(self.robot, self.robot.joint_groups[group])

    def set_group_conf(self, group, positions):
        pbu.set_joint_positions(self.robot, self.get_group_joints(group), positions)

    def set_group_positions(self, group_positions):
        for group, positions in group_positions.items():
            self.set_group_conf(group, positions)

    @property
    def joint_positions(self):
        joints = pbu.get_joints(self.robot, client=self.client)
        joint_positions = pbu.get_joint_positions(
            self.robot, joints, client=self.client
        )
        joint_names = pbu.get_joint_names(self.robot, joints, client=self.client)
        return {k: v for k, v in zip(joint_names, joint_positions)}

    def command_group(self, group, positions, **kwargs):  # TODO: default timeout
        self.set_group_positions({group: positions})

    def command_group_dict(
        self, group, positions_dict, **kwargs
    ):  # TODO: default timeout
        positions = [positions_dict[nm] for nm in self.robot.joint_groups[group]]
        self.command_group(group, positions)

    def command_group_trajectory(
        self, group, positions, times_from_start, dt=0.01, **kwargs
    ):
        for position in positions:
            self.command_group(group, position)
            time.sleep(dt)
            yield

    def wait(self, duration):
        time.sleep(duration)

    def wait_for_clients(self, clients, timeout=0):
        pass

    def any_arm_fully_closed(self):
        return False
