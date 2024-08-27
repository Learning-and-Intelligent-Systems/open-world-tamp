import zlib

import numpy as np
import pickle5
import pyrealsense2 as rs
import rospy
import zmq
from franka_interface import ArmInterface, GripperInterface
from rs_util import get_intrinsics, get_serial_number

rospy.init_node("panda_data_collection_node")

##### Realsense ######

print("Listing available realsense devices...")
serial_numbers = []
datas = []
for i, device in enumerate(rs.context().devices):
    serial_number = device.get_info(rs.camera_info.serial_number)
    serial_numbers.append(serial_number)


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


class CaptureRS:
    def __init__(
        self, callback=None, vis=False, serial_number=None, intrinsics=None, min_tags=1
    ):
        self.callback = callback
        self.vis = vis
        self.min_tags = min_tags

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)

        # Start streaming
        pipeline_profile = self.pipeline.start(config)

        # And get the device info
        self.serial_number = get_serial_number(pipeline_profile)
        print(f"Connected to {self.serial_number}")

        # get the camera intrinsics
        # print('Using default intrinsics from camera.')
        if intrinsics is None:
            self.intrinsics = get_intrinsics(pipeline_profile)
        else:
            self.intrinsics = intrinsics

    def capture(self):
        for _ in range(100):
            # Wait for a coherent pair of frames: depth and color
            frameset = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(frameset.get_color_frame().get_data())

        return color_image, depth_image

    def close(self):
        self.pipeline.stop()


arm_interface = ArmInterface()
gripper_interface = GripperInterface()


def capture_realsense(message):
    bpe = CaptureRS(serial_number=serial_numbers[0])
    (rgb, depth), intrinsics = bpe.capture(), bpe.intrinsics
    message = {"rgb": rgb, "depth": depth, "intrinsics": intrinsics}
    socket.send(zlib.compress(pickle5.dumps(message)))


def command_arm(message):
    arm_interface.move_to_joint_positions(message["positions"])
    message = {"success": True}
    socket.send(zlib.compress(pickle5.dumps(message)))


def get_joint_states(message):
    joints = arm_interface.joint_angles()
    message = {"joint_states": joints}
    socket.send(zlib.compress(pickle5.dumps(message)))


def open_gripper(message):
    gripper_interface.open()
    message = {"success": True}
    socket.send(zlib.compress(pickle5.dumps(message)))


def close_gripper(message):
    gripper_interface.close()
    message = {"success": True}
    socket.send(zlib.compress(pickle5.dumps(message)))


def execute_position_path(message):
    arm_interface.execute_position_path(message["pdicts"])
    message = {"success": True}
    socket.send(zlib.compress(pickle5.dumps(message)))


while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))

    print("Received request: {}".format(message))

    #  Send reply back to client
    globals()[message["message_name"]](message)
