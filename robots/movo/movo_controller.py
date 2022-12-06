import time

from pybullet_tools.utils import joints_from_names, set_joint_positions

from robots.movo.movo_sender import (
    command,
    command_gripper,
    command_torso,
    command_trajectory,
    get_joint_states,
)
from open_world.simulation.controller import SimulatedController
from open_world.estimation.dnn import str_from_int_seg_general
from open_world.estimation.geometry import cloud_from_depth
from open_world.estimation.observation import image_from_labeled, save_camera_images
from pybullet_tools.utils import (
    CameraImage,
    set_joint_positions,
)

class SimulatedMovoController(SimulatedController):
    def __init__(self, robot, verbose=True, **kwargs):
        super(SimulatedMovoController, self).__init__(robot, **kwargs)


class MovoController(object):
    def __init__(self, args, robot, verbose=True, **kwargs):
        self.robot = robot
        self.args=args
        super(MovoController, self).__init__()

    def simulate_trajectory(self, positions, joint_names):
        print(joint_names, positions)
        joints = joints_from_names(self.robot, joint_names)
        for position in positions:
            set_joint_positions(self.robot, joints, position)
            time.sleep(0.02)

    def command_group_trajectory(
        self, group, positions, times_from_start, blocking=True, simulate=True, **kwargs
    ):

        joint_names = self.robot.command_joint_groups[group]

        timeout = 1 / 10.0 * len(positions) * 1.1
        print("Group: " + str(group))
        print("Trajectory length: " + str(len(positions)))
        if "gripper" in group:
            if group == "right_gripper":
                pdict = {name: pos for name, pos in zip(joint_names, positions[-1])}
                self.command_group_dict(group, pdict)

        elif "torso" in group:
            raise NotImplementedError
        else:
            if simulate:
                self.simulate_trajectory(positions, joint_names)
            command_trajectory(group, timeout, joint_names, positions)

    def any_arm_closed(self):
        raise NotImplementedError

    def wait(self, duration=0):
        raise NotImplementedError

    @property
    def joint_names(self):
        raise NotImplementedError

    @property
    def joint_positions(self):
        positions = {}
        joint_states = get_joint_states()
        for group_name, joint_names in self.robot.command_joint_groups.items():
            if group_name != "base":
                current_angles = {
                    joint_name: joint_states["joint_dict"][joint_name]
                    for joint_name in joint_names
                }
                positions.update(current_angles)
        return positions

    def any_arm_fully_closed(self):
        # return self.is_closed
        return False

    @property
    def joint_velocities(self):
        raise NotImplementedError

    def wait_for_clients(self, clients, timeout=0):
        raise NotImplementedError

    def command_arm(self, positions):
        raise NotImplementedError

    def open_gripper(self, arm, blocking=True):
        raise NotImplementedError

    def close_gripper(self, arm, blocking=True):
        raise NotImplementedError

    def wait(self, duration=0):
        time.sleep(duration)

    @property
    def is_open(self):
        raise NotImplementedError

    def get_segmented_image(self, seg_network):
        # Get the point cloud
        rgb_image, depth_image = self.robot.read_images()

        # Filter max depth
        max_depth = 1.5
        depth_image[depth_image > max_depth] = 0

        point_cloud = cloud_from_depth(
            self.robot.intrinsics, depth_image, max_depth=float("inf")
        )
        #  print(rgb_image.shape, depth_image.shape, point_cloud.shape)

        int_seg = seg_network.get_seg(
            rgb_image, point_cloud=point_cloud, return_int=True, depth_image=depth_image
        )
        str_seg = str_from_int_seg_general(
            int_seg, use_classifer=self.args.fasterrcnn_detection
        )
        color_seg = image_from_labeled(str_seg)
        camera_pose = self.robot.get_camera_pose()
        camera_image = CameraImage(
            rgb_image, depth_image, str_seg, camera_pose, self.robot.intrinsics
        )  # TODO: camera_pose
        save_camera_images(camera_image)
        return camera_image

    def command_group_dict(
        self, group, positions, timeout=10.0, start_joint_positions=None
    ):
        if "gripper" in group:
            if group == "right_gripper":
                if list(positions.values())[0] == 0:
                    position = 0.085
                else:
                    position = 0
                command_gripper(group, timeout, position)
        elif "torso" in group:
            start_joint_positions = [
                self.joint_positions[name]
                for name in self.robot.command_joint_groups[group]
            ]
            joint_names = self.robot.command_joint_groups[group]
            goal_joint_positions = [
                positions[name] for name in self.robot.command_joint_groups[group]
            ]
            command_torso(
                group, timeout, joint_names, start_joint_positions, goal_joint_positions
            )
        else:
            start_joint_positions = [
                self.joint_positions[name]
                for name in self.robot.command_joint_groups[group]
            ]
            joint_names = self.robot.command_joint_groups[group]
            goal_joint_positions = [
                positions[name] for name in self.robot.command_joint_groups[group]
            ]
            command(
                group, timeout, joint_names, start_joint_positions, goal_joint_positions
            )
