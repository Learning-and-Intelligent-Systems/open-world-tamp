#!/usr/bin/env python3

from __future__ import print_function

import argparse
import math
import os
import pdb
import pickle
import sys
import time

import mayavi.mlab as mlab
import numpy as np
import pybullet
import pybullet as p
import torch
from minio import Minio
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R

np.set_printoptions(
    precision=3, threshold=3, edgeitems=1, suppress=True
)  # , linewidth=1000)

MODEL_PATH = "./models"
LTAMP_PR2 = os.path.join(MODEL_PATH, "pr2_description/pr2.urdf")
SUGAR_PATH = os.path.join(MODEL_PATH, "ycb/003_cracker_box/textured.obj")
GRIPPER_TEMPLATE = os.path.join(MODEL_PATH, "panda_gripper.obj")

# NOTE(caelan): must come before other imports
sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
        #'pddlstream/examples/pybullet/utils',
    ]
)

import copy
import random
from collections import namedtuple

from pybullet_tools.ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
from pybullet_tools.pr2_primitives import (
    Commands,
    Grasp,
    State,
    apply_commands,
    create_trajectory,
)
from pybullet_tools.pr2_problems import create_table
from pybullet_tools.pr2_utils import (
    ARM_NAMES,
    DRAKE_PR2_URDF,
    GET_GRASPS,
    PR2_CAMERA_MATRIX,
    PR2_GRIPPER_ROOTS,
    PR2_GROUPS,
    PR2_TOOL_FRAMES,
    REST_LEFT_ARM,
    SIDE_HOLDING_LEFT_ARM,
    TOP_HOLDING_LEFT_ARM,
    arm_conf,
    attach_viewcone,
    compute_grasp_width,
    get_arm_joints,
    get_carry_conf,
    get_gripper_joints,
    get_gripper_link,
    get_group_conf,
    get_group_joints,
    get_pr2_field_of_view,
    get_side_grasps,
    get_top_grasps,
    get_viewcone,
    get_x_presses,
    is_drake_pr2,
    joints_from_names,
    learned_pose_generator,
    open_arm,
    pixel_from_ray,
    ray_from_pixel,
    rightarm_from_leftarm,
)
from pybullet_tools.utils import (
    BASE_LINK,
    BLACK,
    GREY,
    INF,
    PI,
    RED,
    TAN,
    TEMP_DIR,
    WHITE,
    Attachment,
    BodySaver,
    HideOutput,
    Pose,
    aabb_from_points,
    add_data_path,
    add_fixed_constraint,
    add_line,
    add_segments,
    all_between,
    apply_alpha,
    base_values_from_pose,
    body_collision,
    connect,
    create_attachment,
    create_mesh,
    create_obj,
    create_plane,
    disable_real_time,
    disconnect,
    draw_aabb,
    draw_mesh,
    draw_point,
    draw_pose,
    elapsed_time,
    enable_gravity,
    euler_from_quat,
    flatten_links,
    get_aabb,
    get_aabb_center,
    get_bodies,
    get_body_name,
    get_camera,
    get_custom_limits,
    get_distance,
    get_extend_fn,
    get_image,
    get_joint_positions,
    get_link_pose,
    get_max_limit,
    get_min_limit,
    get_moving_links,
    get_name,
    get_pose,
    get_unit_vector,
    has_gui,
    image_from_segmented,
    interpolate_poses,
    invert,
    is_placement,
    joint_controller_hold,
    joints_from_names,
    link_from_name,
    load_model,
    load_pybullet,
    mesh_from_points,
    multiply,
    pairwise_collision,
    plan_base_motion,
    plan_direct_joint_motion,
    plan_joint_motion,
    point_from_pose,
    pose_from_base_values,
    pose_from_tform,
    remove_body,
    remove_debug,
    remove_fixed_constraint,
    sample_placement,
    save_image,
    set_base_values,
    set_camera,
    set_camera_pose,
    set_camera_pose2,
    set_joint_positions,
    set_point,
    set_pose,
    set_renderer,
    stable_z,
    step_simulation,
    sub_inverse_kinematics,
    tform_point,
    uniform_pose_generator,
    unit_quat,
    user_input,
    wait_for_duration,
    wait_for_user,
    wait_if_gui,
    waypoints_from_path,
)
from pybullet_tools.voxels import MAX_PIXEL_VALUE

# from grasp.graspnet_interface import visualize_grasps

# from examples.test_visiblity import *
# from learn_tools.collectors.collect_stir import *


# https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pointCloudFromCameraImage.py
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getCameraImageTest.py
# https://www.geeksforgeeks.org/python-pil-image-save-method/


# plot grasps in mayavi
minioClient = Minio(
    "ceph.csail.mit.edu",
    access_key=os.environ["S3ACCESS"],
    secret_key=os.environ["S3SECRET"],
    secure=True,
)


#######################################################

LIS_PR2 = True
if LIS_PR2:
    CAMERA_FRAME = "head_mount_kinect_rgb_link"
    CAMERA_OPTICAL_FRAME = "head_mount_kinect_rgb_optical_frame"
    WIDTH, HEIGHT = 640, 480
    FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
    # INTRINSICS = get_camera_matrix(WIDTH, HEIGHT, FX, FY)
    INTRINSICS = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
else:
    CAMERA_FRAME = "high_def_frame"
    CAMERA_OPTICAL_FRAME = "high_def_optical_frame"  # HEAD_LINK_NAME
    INTRINSICS = PR2_CAMERA_MATRIX
    # TODO: WIDTH, HEIGHT

#######################################################

LabeledPoint = namedtuple("LabledPoint", ["point", "color", "body"])


def set_group_positions(pr2, group_positions):
    for group, positions in group_positions.items():
        joints = joints_from_names(pr2, PR2_GROUPS[group])
        assert len(joints) == len(positions)
        set_joint_positions(pr2, joints, positions)


#######################################################


def convexify(labeled_points, points_to_add=10000):
    new_labeled_points = copy.deepcopy(labeled_points)
    for _ in range(points_to_add):
        pntidx = random.sample(labeled_points, 2)
        t = random.uniform(0, 1)
        new_point = tuple(
            [pntidx[0].point[i] * t + pntidx[1].point[i] * (1 - t) for i in range(3)]
        )
        new_labeled_points.append(
            LabeledPoint(new_point, pntidx[0].color, pntidx[0].body)
        )
    return new_labeled_points


def apply_physical_commands(state, commands, time_step=None, **kwargs):
    for i, command in enumerate(commands):
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            else:
                wait_for_duration(time_step)
                p.stepSimulation()


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--cfree', action='store_true',
    #                     help='When enabled, disables collision checking (for debugging).')
    # parser.add_argument('-p', '--problem', default='test_pour', choices=sorted(problem_fn_from_name),
    #                     help='The name of the problem to solve.')
    # parser.add_argument('-s', '--seed', default=None,
    #                     help='The random seed to use.')
    parser.add_argument("-v", "--viewer", action="store_true", help="")
    args = parser.parse_args()

    connect(use_gui=True)
    draw_pose(Pose(), length=1)
    with HideOutput():  # enable=None):
        add_data_path()
        # floor = create_floor()
        floor = create_plane(color=TAN)
        table = create_table(leg_color=GREY, cylinder=False)
        # table = load_pybullet(TABLE_URDF)
        obj = create_obj(SUGAR_PATH, color=WHITE, mass=0.1)  # , **kwargs)

        if LIS_PR2:
            pr2 = load_pybullet(LTAMP_PR2)
        else:
            pr2 = load_model(DRAKE_PR2_URDF)

    set_pose(obj, ([0, 0, stable_z(obj, table)], p.getQuaternionFromEuler([0, 0, 1])))
    # dump_body(pr2)
    group_positions = {
        "base": [-0.5, 0, 0],
        "left_arm": REST_LEFT_ARM,
        "right_arm": rightarm_from_leftarm(REST_LEFT_ARM),
        "torso": [0.2],
        "head": [0, PI / 4],
    }
    set_group_positions(pr2, group_positions)
    # wait_if_gui()

    table_aabb = get_aabb(table)
    draw_aabb(table_aabb)
    x, y, _ = get_aabb_center(table_aabb)
    # target_point = [x, y, table_aabb[1][2]]

    camera_matrix = INTRINSICS
    camera_link = link_from_name(pr2, CAMERA_OPTICAL_FRAME)
    camera_pose = get_link_pose(pr2, camera_link)
    draw_pose(camera_pose, length=1)  # TODO: attach to camera_link

    distance = 2.5
    target_point = tform_point(camera_pose, np.array([0, 0, distance]))
    attach_viewcone(
        pr2,
        depth=distance,
        head_name=CAMERA_FRAME,
        camera_matrix=camera_matrix,
        color=RED,
    )
    # view_cone = get_viewcone(depth=distance, camera_matrix=camera_matrix, color=apply_alpha(RED, alpha=0.1))
    # set_pose(view_cone, camera_pose)

    # set_camera_pose2(camera_pose, distance)
    # camera_info = get_camera()
    # wait_if_gui()
    # return

    # TODO: OpenCV
    camera_point = point_from_pose(camera_pose)
    _, vertical_fov = get_pr2_field_of_view(camera_matrix=camera_matrix)
    # view_matrix = p.computeViewMatrix(cameraEyePosition=camera_point, cameraTargetPosition=target_point,
    #                                  cameraUpVector=[0, 0, 1]) #, physicsClientId=CLIENT)
    # view_pose = pose_from_tform(np.reshape(view_matrix, [4, 4]))
    # projection_matrix = get_projection_matrix(width, height, vertical_fov, near, far)

    rgb_image, depth_image, seg_image = get_image(
        camera_pos=camera_point,
        target_pos=target_point,
        width=WIDTH,
        height=HEIGHT,
        vertical_fov=vertical_fov,
        tiny=False,
        segment=True,
        segment_links=False,
    )

    save_image(os.path.join(TEMP_DIR, "rgb.png"), rgb_image)  # [0, 255]
    save_image(os.path.join(TEMP_DIR, "depth.png"), depth_image)  # [0, 1]
    if seg_image is not None:
        segmented_image = image_from_segmented(seg_image, color_from_body=None)
        save_image(os.path.join(TEMP_DIR, "segmented.png"), segmented_image)  # [0, 1]

    step_size = 1
    labeled_points = []
    for r in range(0, depth_image.shape[0], step_size):
        for c in range(0, depth_image.shape[1], step_size):
            body, link = seg_image[r, c, :]
            if body not in [obj]:
                continue
            pixel = [c, r]  # NOTE: width, height
            ray = ray_from_pixel(camera_matrix, pixel)
            depth = depth_image[r, c]
            point_camera = depth * ray
            point_world = tform_point(multiply(camera_pose), point_camera)
            point_world = [point_world[1], point_world[2], point_world[0]]
            color = rgb_image[r, c, :] / MAX_PIXEL_VALUE
            labeled_points.append(LabeledPoint(point_world, color, body))
            # draw_point(point_world, size=0.01, color=color) # TODO: adjust size based on step_size
            # add_line(camera_point, point_world, color=color)

    labeled_points = convexify(labeled_points)
    points = [point.point for point in labeled_points]
    mean_color = np.average(
        [point.color for point in labeled_points], axis=0
    )  # average | median
    centroid = np.average(points, axis=0)
    obj_aabb = aabb_from_points(points)
    origin_point = centroid
    origin_point[2] = obj_aabb[0][2]
    origin_pose = Pose(origin_point)  # PCA
    draw_pose(origin_pose)

    # draw_aabb(obj_aabb, color=mean_color)
    obj_mesh = mesh_from_points(points)
    # draw_mesh(obj_mesh, color=mean_color)
    obj_approx = create_mesh(obj_mesh, under=True, color=mean_color)

    # TODO: simplify meshes

    # plot the full 3d point cloud in mayavi
    pc = np.array([p.point for p in labeled_points])
    pc_color = np.array([p.color for p in labeled_points])[:, :3] * 255

    # # Read the data from minio and display it in mayavi

    def read_from_minio(file):
        tmp_folder = "./tmp_minio/"
        if not os.path.isdir(tmp_folder):
            os.mkdir(tmp_folder)

        tmp_path = tmp_folder + file

        # s to get objects in folder
        minioClient.fget_object("aidan_bucket", file, tmp_path)

        # Write data to a pickle file
        with open(tmp_path, "rb") as handle:
            b = pickle.load(handle)
        return b

    data = read_from_minio("mayavi_data_1603094225.8715823.pkl")
    np.set_printoptions(threshold=sys.maxsize)

    def flip_rot(rot):
        nrot = rot + math.pi
        if nrot > 2 * math.pi:
            return nrot - 2 * math.pi
        return nrot

    # Load in the URDF as an object
    templates = []
    grasp_transforms = []
    orig_grasp_index = []
    for gi, grasp in enumerate(data["grasps"]):
        gripper_template = create_obj(GRIPPER_TEMPLATE, color=WHITE)
        position_vec = grasp[:3, 3]
        position_vec = [position_vec[2], position_vec[0], position_vec[1]]

        p1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        # position_vec = [position_vec[2], position_vec[0], position_vec[1]]
        r = R.from_matrix(np.matmul(p1, grasp[:3, :3]))
        roll, pitch, yaw = list(r.as_euler("xyz", degrees=False))

        set_pose(gripper_template, Pose(position_vec, [roll, pitch, yaw]))
        if body_collision(gripper_template, table):
            remove_body(gripper_template)
        else:
            pre_translate_matrix = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]]
            )
            double_transform_matrix = np.matmul(grasp, pre_translate_matrix)
            gripper_position_vec = double_transform_matrix[:3, 3]
            gripper_position_vec = [
                gripper_position_vec[2],
                gripper_position_vec[0],
                gripper_position_vec[1],
            ]
            orig_grasp_index.append(gi)
            gr = R.from_matrix(np.matmul(p1, double_transform_matrix[:3, :3]))
            groll, gpitch, gyaw = list(gr.as_euler("xyz", degrees=False))
            gquat = p.getQuaternionFromEuler(
                [groll + math.pi / 2.0, gpitch + math.pi / 2.0, gyaw]
            )
            grasp_transforms.append((tuple(gripper_position_vec), gquat))

            templates.append(gripper_template)
        # set_pose(gripper_template, Pose(position_vec, [roll, pitch, yaw]))

    # grasp_index = random.randint(0, len(grasp_transforms))
    grasp_index = 57
    remaining = None
    for ti, template in enumerate(templates):
        if ti != grasp_index:
            remove_body(template)
        else:
            remaining = template

    # Get the agent to actually grasp the object with the grasp transpose

    arm = ARM_NAMES[0]
    GRASP_LENGTH = 0.03
    APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
    approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0])

    grasp_targets = [
        ((g[0][0], g[0][1], g[0][2] + 0.1), (g[1][0], g[1][1], g[1][2], g[1][3]))
        for g in [grasp_transforms[grasp_index]]
    ]
    grasps = []
    grasps.extend(
        Grasp(
            "top",
            obj,
            g,
            multiply((approach_vector, unit_quat()), g),
            TOP_HOLDING_LEFT_ARM,
        )
        for g in [grasp_transforms[grasp_index]]
    )

    arm_link = get_gripper_link(pr2, arm)
    arm_joints = get_arm_joints(pr2, arm)

    pose = get_pose(obj)
    obstacles = []
    approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}

    grasp = grasps[0]
    gripper_pose = grasp.value  # w_f_g = w_f_o * (g_f_o)^-1
    approach_pose = grasp.approach
    default_conf = arm_conf(arm, grasp.carry)
    # sample_fn = get_sample_fn(robot, arm_joints)
    # base_conf.assign()
    custom_limits = {}
    open_arm(pr2, arm)
    set_joint_positions(pr2, arm_joints, default_conf)  # default_conf | sample_fn()

    print(pr2, arm, gripper_pose)
    grasp_conf = pr2_inverse_kinematics(
        pr2, arm, gripper_pose, custom_limits=custom_limits
    )  # , upper_limits=USE_CURRENT)

    approach_conf = sub_inverse_kinematics(
        pr2, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits
    )
    approach_conf = get_joint_positions(pr2, arm_joints)

    resolutions = 0.01 ** np.ones(len(arm_joints))
    grasp_path = plan_direct_joint_motion(
        pr2,
        arm_joints,
        grasp_conf,
        obstacles=approach_obstacles,
        self_collisions=False,
        resolutions=resolutions / 2.0,
        attachments=[],
    )
    if grasp_path is None:
        return None
    set_joint_positions(pr2, arm_joints, default_conf)
    approach_path = plan_joint_motion(
        pr2,
        arm_joints,
        approach_conf,
        obstacles=obstacles,
        self_collisions=False,
        resolutions=resolutions,
        restarts=2,
        iterations=25,
        smooth=25,
    )
    if approach_path is None:
        print("Approach path failure")
        return None

    # TARGET POSITION
    target_grasps = []
    target_grasps.extend(
        Grasp(
            "top",
            obj,
            g,
            multiply((approach_vector, unit_quat()), g),
            TOP_HOLDING_LEFT_ARM,
        )
        for g in grasp_targets
    )

    target_grasp = target_grasps[0]
    target_gripper_pose = target_grasp.value  # w_f_g = w_f_o * (g_f_o)^-1
    target_approach_pose = target_grasp.approach

    set_joint_positions(pr2, arm_joints, grasp_conf)  # default_conf | sample_fn()

    target_grasp_conf = pr2_inverse_kinematics(
        pr2, arm, target_gripper_pose, custom_limits=custom_limits
    )  # , upper_limits=USE_CURRENT)
    target_approach_conf = sub_inverse_kinematics(
        pr2, arm_joints[0], arm_link, target_approach_pose, custom_limits=custom_limits
    )
    target_approach_conf = get_joint_positions(pr2, arm_joints)

    resolutions = 0.01 ** np.ones(len(arm_joints))

    target_grasp_path = plan_direct_joint_motion(
        pr2,
        arm_joints,
        target_grasp_conf,
        obstacles=approach_obstacles,
        self_collisions=False,
        resolutions=resolutions / 2.0,
        attachments=[],
    )

    set_joint_positions(pr2, arm_joints, grasp_conf)
    target_approach_path = plan_joint_motion(
        pr2,
        arm_joints,
        target_approach_conf,
        obstacles=obstacles,
        self_collisions=False,
        resolutions=resolutions,
        restarts=2,
        iterations=25,
        smooth=25,
    )
    if approach_path is None:
        return None

    path = approach_path + grasp_path
    pre_mt = create_trajectory(pr2, arm_joints, path)
    pre_grasp_cmd = Commands(State(), savers=[BodySaver(pr2)], commands=[pre_mt])

    target_path = target_approach_path
    post_mt = create_trajectory(pr2, arm_joints, target_path)
    post_grasp_cmd = Commands(State(), savers=[BodySaver(pr2)], commands=[post_mt])
    # pdb.set_trace()
    remove_body(remaining)
    apply_commands(State(), pre_grasp_cmd.commands, time_step=0.01)
    link = link_from_name(pr2, PR2_TOOL_FRAMES.get(arm, arm))
    add_fixed_constraint(obj, pr2, link)

    apply_physical_commands(State(), post_grasp_cmd.commands, time_step=0.01)
    for _ in range(500):
        p.stepSimulation()
        time.sleep(0.01)
    wait_if_gui()

    disconnect()


if __name__ == "__main__":
    main()
