import os

import pybullet as p

import owt.pb_utils as pbu

MOVO_URDF = "models/srl/movo_description/movo_robotiq_collision.urdf"
MOVO_PATH = os.path.abspath(MOVO_URDF)


LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_arm_half_joint",
    "left_elbow_joint",
    "left_wrist_spherical_1_joint",
    "left_wrist_spherical_2_joint",
    "left_wrist_3_joint",
    "linear_joint",
]


RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_arm_half_joint",
    "right_elbow_joint",
    "right_wrist_spherical_1_joint",
    "right_wrist_spherical_2_joint",
    "right_wrist_3_joint",
    "linear_joint",
]

RIGHT_ATTACH_CONF = [
    -0.021811289327748895,
    -0.5591495793058756,
    0.09515283160149757,
    -0.9770537496674913,
    0.22921576166484137,
    1.059975131790689,
    -1.6935222466767996,
    0,
]
LEFT_ATTACH_CONF = [
    -0.2760957691629127,
    0.5009078441624968,
    0.2956304885223213,
    1.2349056669408707,
    -0.012336294801464476,
    -0.3835782875974208,
    1.7257314490066005,
    0,
]

TUCKED_DEFAULT_JOINTS = {
    "pan_joint": -0.07204942405223846,
    "tilt_joint": -0.799216890335083,
    "left_shoulder_pan_joint": 1.6,
    "left_shoulder_lift_joint": 1.4,
    "left_arm_half_joint": -0.4,
    "left_elbow_joint": 2.7,
    "left_wrist_spherical_1_joint": 0.0,
    "left_wrist_spherical_2_joint": -0.5,
    "left_wrist_3_joint": 1.7,
    "right_shoulder_pan_joint": -1.6,
    "right_shoulder_lift_joint": -1.4,
    "right_arm_half_joint": 0.4,
    "right_elbow_joint": -2.7,
    "right_wrist_spherical_1_joint": 0.0,
    "right_wrist_spherical_2_joint": 0.5,
    "right_wrist_3_joint": -1.7,
    "left_gripper_finger1_joint": -0.0008499202079690222,
    "left_gripper_finger2_joint": -0.0,
    "left_gripper_finger3_joint": 0.0,
    "right_gripper_finger1_joint": 0.0,
    "linear_joint": 0.04,
    "right_gripper_finger1_joint": 0,
    "right_gripper_finger2_joint": 0,
    "right_gripper_finger1_inner_knuckle_joint": 0,
    "right_gripper_finger2_inner_knuckle_joint": 0,
    "right_gripper_finger1_finger_tip_joint": 0,
    "right_gripper_finger2_finger_tip_joint": 0,
}


DEFAULT_JOINTS = {
    "pan_joint": -0.07204942405223846,
    "tilt_joint": -0.799216890335083,
    "left_shoulder_pan_joint": 1.6,
    "left_shoulder_lift_joint": 1.6,
    "left_arm_half_joint": -2.9308729618144724,
    "left_elbow_joint": 0,
    "left_wrist_spherical_1_joint": -0.4464001271835176,
    "left_wrist_spherical_2_joint": 2.8,
    "left_wrist_3_joint": 1.859177258066345,
    "right_shoulder_pan_joint": -1.6,
    "right_shoulder_lift_joint": -1.6,
    "right_arm_half_joint": -0.3052025219808243,
    "right_elbow_joint": 0,
    "right_wrist_spherical_1_joint": 0.5568418170632672,
    "right_wrist_spherical_2_joint": 2.8,
    "right_wrist_3_joint": -0.059844425487387554,
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


def check_initial_end(start_conf, end_conf, collision_fn, verbose=True):
    # TODO: collision_fn might not accept kwargs
    if collision_fn(start_conf, verbose=verbose):
        print("Warning: initial configuration is in collision")
        return False
    if collision_fn(end_conf, verbose=verbose):
        print("Warning: end configuration is in collision")
        return False
    return True


def create_pillar(width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
    # TODO: use the color when naming
    return pbu.create_box(w=width, l=length, h=height, color=color, **kwargs)


def setup_robot_pybullet():
    p.connect(p.DIRECT)
    robot_body = pbu.load_pybullet(MOVO_PATH, fixed_base=True)
    return robot_body
