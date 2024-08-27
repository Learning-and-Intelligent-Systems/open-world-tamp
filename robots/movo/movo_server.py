import ctypes
import functools
import struct
import threading
import zlib

import actionlib
import numpy as np
import pickle5
import rospy
import sensor_msgs.point_cloud2 as pc2
import zmq
from control_msgs.msg import (FollowJointTrajectoryAction,
                              FollowJointTrajectoryGoal,
                              JointTrajectoryControllerState)
from movo_msgs.msg import GripperCmd, GripperStat
from rtabmap_ros.msg import MapData
from sensor_msgs.msg import Image, JointState, PointCloud2, PointField
from trajectory_msgs.msg import JointTrajectoryPoint

rospy.init_node("M0M")
rospy.sleep(
    1
)  # NOTE Wait for the subscriber to get some initial data. else read rgb returns None


torso_client = actionlib.SimpleActionClient(
    "movo/torso_controller/follow_joint_trajectory",
    FollowJointTrajectoryAction,
)

left_arm_client = actionlib.SimpleActionClient(
    "movo/left_arm_controller/follow_joint_trajectory",
    FollowJointTrajectoryAction,
)

right_arm_client = actionlib.SimpleActionClient(
    "movo/right_arm_controller/follow_joint_trajectory",
    FollowJointTrajectoryAction,
)

left_gripper_client = rospy.Publisher(
    "movo/left_gripper/cmd", GripperCmd, queue_size=10
)

right_gripper_client = rospy.Publisher(
    "movo/right_gripper/cmd", GripperCmd, queue_size=10
)

head_client = actionlib.SimpleActionClient(
    "movo/head_controller/follow_joint_trajectory",
    FollowJointTrajectoryAction,
)


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


def color_image_callback(event, rgb):
    if not event.is_set():
        message = {"height": rgb.height, "width": rgb.width, "data": rgb.data}
        socket.send(zlib.compress(pickle5.dumps({"color_image": message})))
        event.set()
    return True


def get_color_image(message):
    print("Getting color image")
    event = threading.Event()
    callback_with_event = functools.partial(color_image_callback, event)
    rospy.Subscriber("/movo_camera/color/image_color_rect", Image, callback_with_event)
    event.wait()


def depth_image_callback(event, depth):
    if not event.is_set():
        message = {"height": depth.height, "width": depth.width, "data": depth.data}
        socket.send(zlib.compress(pickle5.dumps({"depth_image": message})))
        event.set()
    return True


def pointcloud_callback(event, pc):
    if not event.is_set():
        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])
        # self.lock.acquire()
        gen = pc2.read_points(pc, skip_nans=True)
        int_data = list(gen)
        print("Proc")
        # for xi, x in enumerate(int_data):
        #     print("{}/{}".format(xi, len(int_data)))
        #     test = x[3]
        #     # cast float32 to int so that bitwise operations are possible
        #     s = struct.pack('>f' ,test)
        #     i = struct.unpack('>l',s)[0]
        #     # you can get back the float value by the inverse operations
        #     pack = ctypes.c_uint32(i).value
        #     r = (pack & 0x00FF0000)>> 16
        #     g = (pack & 0x0000FF00)>> 8
        #     b = (pack & 0x000000FF)
        #     # prints r,g,b values in the 0-255 range
        #                 # x,y,z can be retrieved from the x[0],x[1],x[2]
        #     xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
        #     rgb = np.append(rgb,[[r,g,b]], axis = 0)

        message = {"int_data": int_data}
        # message = {"xyz":xyz, "rgb": rgb}
        socket.send(zlib.compress(pickle5.dumps(message)))
        event.set()


def get_pointcloud(message):
    event = threading.Event()
    callback_with_event = functools.partial(pointcloud_callback, event)
    rospy.Subscriber("/rtabmap/cloud_map", PointCloud2, callback_with_event)
    event.wait()


def slam_graph_callback(event, map_data):
    if not event.is_set():
        message = {"map_data": map_data}
        socket.send(zlib.compress(pickle5.dumps(message)))
        event.set()


def get_slam_graph(message):
    event = threading.Event()
    callback_with_event = functools.partial(pointcloud_callback, event)
    rospy.Subscriber("/rtabmap/mapData", MapData, callback_with_event)
    event.wait()


def get_depth_image(message):
    event = threading.Event()
    callback_with_event = functools.partial(depth_image_callback, event)
    rospy.Subscriber("/movo_camera/depth/image_depth_rect", Image, callback_with_event)
    event.wait()


def get_joint_states(message):
    joint_val = rospy.wait_for_message("/joint_states", JointState)
    message = {
        "joint_dict": {
            name: position for name, position in zip(joint_val.name, joint_val.position)
        }
    }
    socket.send(zlib.compress(pickle5.dumps(message)))


def get_command_goal(message):
    goal = FollowJointTrajectoryGoal()
    goal.goal_time_tolerance = rospy.Time(0.1)
    goal.trajectory.joint_names = message["joint_names"]

    # Add current pose as start ponit
    start_goal_point = JointTrajectoryPoint()
    start_goal_point.positions = message["current_joint_positions"]
    start_goal_point.velocities = [0] * len(goal.trajectory.joint_names)
    start_goal_point.time_from_start = rospy.Duration(0)
    goal.trajectory.points.append(start_goal_point)

    # Add end point
    goal_point = JointTrajectoryPoint()
    goal_point.positions = message["goal_joint_positions"]
    goal_point.velocities = [0] * len(goal.trajectory.joint_names)
    goal_point.time_from_start = rospy.Duration(10)
    goal.trajectory.points.append(goal_point)
    return goal


def get_command_trajectoary(message):
    goal = FollowJointTrajectoryGoal()
    goal.goal_time_tolerance = rospy.Time(0.1)
    goal.trajectory.joint_names = message["joint_names"]

    # Add current pose as start ponit
    for ti, traj_positions in enumerate(message["trajectory_joint_positions"]):
        start_goal_point = JointTrajectoryPoint()
        start_goal_point.positions = traj_positions
        start_goal_point.velocities = [0] * len(goal.trajectory.joint_names)
        start_goal_point.time_from_start = rospy.Duration(ti / 10.0)
        goal.trajectory.points.append(start_goal_point)

    return goal


def command_trajectory_left_arm(message):
    left_arm_client.send_goal(get_command_trajectoary(message))
    left_arm_client.wait_for_result(timeout=rospy.Duration(message["timeout"]))
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_trajectory_right_arm(message):
    right_arm_client.send_goal(get_command_trajectoary(message))
    right_arm_client.wait_for_result(timeout=rospy.Duration(message["timeout"]))
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_left_gripper(message):
    gripper_command = GripperCmd()
    # Closed position: 0.085     open position: 0.0
    gripper_command.position = message["position"]
    gripper_command.speed = 0.02
    gripper_command.force = 100
    left_gripper_client.publish(gripper_command)
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_right_gripper(message):
    gripper_command = GripperCmd()
    # Closed position: 0.085     open position: 0.0
    gripper_command.position = message["position"]
    gripper_command.speed = 0.02
    gripper_command.force = 100
    right_gripper_client.publish(gripper_command)
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_trajectory_head(message):
    head_client.send_goal(get_command_trajectoary(message))
    head_client.wait_for_result(timeout=rospy.Duration(message["timeout"]))
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_left_arm(message):
    left_arm_client.send_goal(get_command_goal(message))
    left_arm_client.wait_for_result(timeout=rospy.Duration(message["timeout"]))
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_right_arm(message):
    right_arm_client.send_goal(get_command_goal(message))
    right_arm_client.wait_for_result(timeout=rospy.Duration(message["timeout"]))
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_torso(message):
    torso_client.send_goal(get_command_goal(message))
    torso_client.wait_for_result(timeout=rospy.Duration(message["timeout"]))
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_head(message):
    head_client.send_goal(get_command_goal(message))
    head_client.wait_for_result(timeout=rospy.Duration(message["timeout"]))
    socket.send(zlib.compress(pickle5.dumps({"success": True})))


def command_base(message):
    print(message)
    socket.send(zlib.compress(pickle5.dumps({"success": False})))


while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))

    print("Received request: {}".format(message))

    #  Send reply back to client
    globals()[message["message_name"]](message)
