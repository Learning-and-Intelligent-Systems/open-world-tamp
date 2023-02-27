
import zlib
import zmq
import pickle5


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

default_joints = {
    "x": 0,
    "y":0,
    "theta":0,
    "pan_joint": -0.07204942405223846,
    "tilt_joint": -0.599216890335083,
    "left_shoulder_pan_joint": 1.0,
    "left_shoulder_lift_joint": 1.9619225455538198,
    "left_arm_half_joint": 0.13184053877842938,
    "left_elbow_joint": 1.8168894557491948,
    "left_wrist_spherical_1_joint": -0.30988063075165684,
    "left_wrist_spherical_2_joint": -1.753361745316172,
    "left_wrist_3_joint": 1.725726522158583,
    "right_shoulder_pan_joint": -1,
    "right_shoulder_lift_joint": -1.9861489225161073,
    "right_arm_half_joint": 0.02609983172656305,
    "right_elbow_joint": -1.8699706504902727,
    "right_wrist_spherical_1_joint": 0.2607507015409034,
    "right_wrist_spherical_2_joint": 1.5755063934988107,
    "right_wrist_3_joint": -1.4726268826923956,
    "left_gripper_finger1_joint": -0.0008499202079690222,
    "left_gripper_finger2_joint": -0.0,
    "left_gripper_finger3_joint": 0.0,
    "right_gripper_finger1_joint": 0.0,
    "linear_joint": 0.3,
    "right_gripper_finger1_joint": 0,
    "right_gripper_finger2_joint": 0,
    "right_gripper_finger1_inner_knuckle_joint": 0,
    "right_gripper_finger2_inner_knuckle_joint": 0,
    "right_gripper_finger1_finger_tip_joint": 0,
    "right_gripper_finger2_finger_tip_joint": 0,
}

def get_joint_states(message):
    # joint_val = rospy.wait_for_message("/joint_states", JointState)
    message = {
        "joint_dict": default_joints
    }
    socket.send(zlib.compress(pickle5.dumps(message)))


def command_base(message):
    print(f"Commanding base with message: {message}")
    socket.send(zlib.compress(pickle5.dumps({"success": True})))
    
while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))

    print("Received request: {}".format(message))

    #  Send reply back to client
    globals()[message["message_name"]](message)
