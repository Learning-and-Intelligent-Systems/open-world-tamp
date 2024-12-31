import ctypes
import pickle
import struct
import zlib

import numpy as np
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.0.246:5555")


def get_pointcloud():
    socket.send(zlib.compress(pickle.dumps({"message_name": "get_pointcloud"})))
    message = pickle.loads(zlib.decompress(socket.recv()))
    xyz = np.array([[0, 0, 0]])
    rgb = np.array([[0, 0, 0]])
    for x in message["int_data"]:
        test = x[3]
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack(">f", test)
        i = struct.unpack(">l", s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = pack & 0x000000FF
        # prints r,g,b values in the 0-255 range
        # x,y,z can be retrieved from the x[0],x[1],x[2]
        xyz = np.append(xyz, [[x[0], x[1], x[2]]], axis=0)
        rgb = np.append(rgb, [[r, g, b]], axis=0)

    return {"xyz": xyz, "rgb": rgb}


def get_color_image():
    socket.send(zlib.compress(pickle.dumps({"message_name": "get_color_image"})))
    message = pickle.loads(zlib.decompress(socket.recv()))["color_image"]
    return message


def get_depth_image():
    socket.send(zlib.compress(pickle.dumps({"message_name": "get_depth_image"})))
    message = pickle.loads(zlib.decompress(socket.recv()))["depth_image"]
    return message


def get_joint_states():
    socket.send(zlib.compress(pickle.dumps({"message_name": "get_joint_states"})))
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message


def command(group, timeout, joint_names, current_joint_positions, goal_joint_positions):
    socket.send(
        zlib.compress(
            pickle.dumps(
                {
                    "message_name": "command_{}".format(group),
                    "timeout": timeout,
                    "joint_names": joint_names,
                    "current_joint_positions": current_joint_positions,
                    "goal_joint_positions": goal_joint_positions,
                }
            )
        )
    )
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message


def command_trajectory(group, timeout, joint_names, trajectory_joint_positions):
    socket.send(
        zlib.compress(
            pickle.dumps(
                {
                    "message_name": "command_trajectory_{}".format(group),
                    "timeout": timeout,
                    "joint_names": joint_names,
                    "trajectory_joint_positions": trajectory_joint_positions,
                }
            )
        )
    )
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message


def command_torso(
    group, timeout, joint_names, current_joint_positions, goal_joint_positions
):
    socket.send(
        zlib.compress(
            pickle.dumps(
                {
                    "message_name": "command_torso",
                    "timeout": timeout,
                    "joint_names": joint_names,
                    "current_joint_positions": current_joint_positions,
                    "goal_joint_positions": goal_joint_positions,
                }
            )
        )
    )
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message


def command_gripper(group, timeout, position):
    socket.send(
        zlib.compress(
            pickle.dumps(
                {
                    "message_name": "command_{}".format(group),
                    "timeout": timeout,
                    "position": position,
                }
            )
        )
    )
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message


def command_base(timeout, position):
    socket.send(
        zlib.compress(
            pickle.dumps(
                {
                    "message_name": "command_base",
                    "timeout": timeout,
                    "position": position,
                }
            )
        )
    )
    message = pickle.loads(zlib.decompress(socket.recv()))
    print("Command base get position")
    return message
