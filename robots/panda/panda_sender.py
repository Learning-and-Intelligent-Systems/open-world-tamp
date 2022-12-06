import pickle
import zlib

import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.0.246:5555")


def capture_realsense():
    socket.send(zlib.compress(pickle.dumps({"message_name": "get_joint_states"})))
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message["rgb"], message["depth"], message["intrinsics"]


def command_arm(positions):
    socket.send(
        zlib.compress(
            pickle.dumps({"message_name": "command_arm", "positions": positions})
        )
    )
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message


def get_joint_states():
    socket.send(zlib.compress(pickle.dumps({"message_name": "get_joint_states"})))
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message["joint_states"]


def open_gripper():
    socket.send(zlib.compress(pickle.dumps({"message_name": "open_gripper"})))
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message


def close_gripper():
    socket.send(zlib.compress(pickle.dumps({"message_name": "close_gripper"})))
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message


def execute_position_path(pdicts):
    socket.send(
        zlib.compress(
            pickle.dumps({"message_name": "execute_position_path", "pdicts": pdicts})
        )
    )
    message = pickle.loads(zlib.decompress(socket.recv()))
    return message
