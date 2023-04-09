#!/usr/bin/env python3

from __future__ import print_function

import argparse
import math
import os
import sys

import mayavi.mlab as mlab
import numpy as np
import pybullet as p

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

from pybullet_tools.pr2_problems import create_table
from pybullet_tools.pr2_utils import (
    DRAKE_PR2_URDF,
    PR2_CAMERA_MATRIX,
    PR2_GROUPS,
    REST_LEFT_ARM,
    attach_viewcone,
    get_pr2_field_of_view,
    get_viewcone,
    pixel_from_ray,
    ray_from_pixel,
    rightarm_from_leftarm,
)
from pybullet_tools.utils import (
    BLACK,
    GREY,
    PI,
    RED,
    TAN,
    TEMP_DIR,
    WHITE,
    HideOutput,
    Pose,
    aabb_from_points,
    add_data_path,
    add_line,
    apply_alpha,
    connect,
    create_mesh,
    create_obj,
    create_plane,
    disconnect,
    draw_aabb,
    draw_mesh,
    draw_point,
    draw_pose,
    get_aabb,
    get_aabb_center,
    get_camera,
    get_image,
    get_link_pose,
    get_pose,
    image_from_segmented,
    invert,
    joints_from_names,
    link_from_name,
    load_model,
    load_pybullet,
    mesh_from_points,
    multiply,
    point_from_pose,
    pose_from_tform,
    save_image,
    set_camera,
    set_camera_pose,
    set_camera_pose2,
    set_joint_positions,
    set_point,
    set_pose,
    stable_z,
    tform_point,
    wait_if_gui,
)
from pybullet_tools.voxels import MAX_PIXEL_VALUE

from grasp.graspnet_interface import (
    generate_demo_grasps,
    score_grasps,
    visualize_grasps,
)

np.set_printoptions(
    precision=3, threshold=3, edgeitems=1, suppress=True
)  # , linewidth=1000)

MODEL_PATH = "./models"
LTAMP_PR2 = os.path.join(MODEL_PATH, "pr2_description/pr2.urdf")
# SUGAR_PATH = os.path.join(MODEL_PATH, 'ycb/003_cracker_box/textured.obj')

obj_name = "024_bowl"
SUGAR_PATH = os.path.join(MODEL_PATH, "ycb/" + obj_name + "/textured.obj")


# object_paths = [
#     'ycb/003_cracker_box/textured.obj',
#     'ycb/004_sugar_box/textured.obj',
#     'ycb/005_tomato_soup_can/textured.obj',
#     'ycb/006_mustard_bottle/textured.obj',
#     'ycb/008_pudding_box/textured.obj',
#     'ycb/009_gelatin_box/textured.obj',
#     'ycb/010_potted_meat_can/textured.obj',
#     'ycb/024_bowl/textured.obj',
# ]

# from examples.test_visiblity import *
# from learn_tools.collectors.collect_stir import *


# https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pointCloudFromCameraImage.py
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getCameraImageTest.py
# https://www.geeksforgeeks.org/python-pil-image-save-method/

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

    connect(use_gui=args.viewer)
    draw_pose(Pose(), length=1)
    with HideOutput():  # enable=None):
        add_data_path()
        # floor = create_floor()
        floor = create_plane(color=TAN)
        table = create_table(leg_color=GREY, cylinder=False)
        # table = load_pybullet(TABLE_URDF)
        obj = create_obj(SUGAR_PATH, color=WHITE)  # , **kwargs)

        if LIS_PR2:
            pr2 = load_pybullet(LTAMP_PR2)
        else:
            pr2 = load_model(DRAKE_PR2_URDF)

    set_pose(obj, ([0, 0, stable_z(obj, table)], p.getQuaternionFromEuler([0, 0, 1])))
    # dump_body(pr2)
    group_positions = {
        "base": [-1.0, 0, 0],
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
    target_point = point_from_pose(get_pose(obj))
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
    cp1 = point_from_pose(camera_pose)
    cp2 = (-cp1[0], -cp1[1], cp1[2])
    cp3 = (-cp1[0], cp1[1], cp1[2])
    cp4 = (cp1[0], -cp1[1], cp1[2])
    camera_points = [cp1, cp2, cp3, cp4]
    labeled_points = []

    for persp_num, camera_point in enumerate(camera_points):

        _, vertical_fov = get_pr2_field_of_view(camera_matrix=camera_matrix)

        camera_image = get_image(
            camera_pos=camera_point,
            target_pos=target_point,
            width=WIDTH,
            height=HEIGHT,
            vertical_fov=vertical_fov,
            tiny=False,
            segment=True,
            segment_links=False,
        )
        rgb_image, depth_image, seg_image = (
            camera_image.rgbPixels,
            camera_image.depthPixels,
            camera_image.segmentationMaskBuffer,
        )

        save_image(
            os.path.join(TEMP_DIR, str(persp_num) + "_rgb.png"), rgb_image
        )  # [0, 255]
        save_image(
            os.path.join(TEMP_DIR, str(persp_num) + "_depth.png"), depth_image
        )  # [0, 1]
        if seg_image is not None:
            segmented_image = image_from_segmented(seg_image, color_from_body=None)
            save_image(
                os.path.join(TEMP_DIR, "segmented.png"), segmented_image
            )  # [0, 1]

        step_size = 1
        for r in range(0, depth_image.shape[0], step_size):
            for c in range(0, depth_image.shape[1], step_size):
                body, link = seg_image[r, c, :]
                if body not in [obj]:
                    continue
                pixel = [c, r]  # NOTE: width, height
                # print(pixel, pixel_from_ray(camera_matrix, ray_from_pixel(camera_matrix, pixel)))
                ray = ray_from_pixel(camera_matrix, pixel)
                depth = depth_image[r, c]
                point_camera = depth * ray
                point_world = tform_point(multiply(camera_pose), point_camera)
                point_world = [point_world[1], point_world[2], point_world[0]]
                color = rgb_image[r, c, :] / MAX_PIXEL_VALUE
                labeled_points.append(LabeledPoint(point_world, color, body))
                # draw_point(point_world, size=0.01, color=color) # TODO: adjust size based on step_size
                # add_line(camera_point, point_world, color=color)

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

    # plot the full 3d point cloud in mayavi
    pc = np.array([p.point for p in labeled_points])
    pc_color = np.array([p.color for p in labeled_points])[:, :3] * 255

    # rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
    # rgba[:, :3] = np.asarray(pc_color)
    # rgba[:, 3] = 255
    # src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    # src.add_attribute(rgba, 'colors')
    # src.data.point_data.set_active_scalars('colors')
    # g = mlab.pipeline.glyph(src)
    # g.glyph.scale_mode = "data_scaling_off"
    # g.glyph.glyph.scale_factor = 0.01
    # mlab.show()

    # Pass through the graspnet for inference
    estimator = visualize_grasps(pc, pc_color, obj_name)
    pcs, demo_grasps = generate_demo_grasps(estimator, pc)
    score_grasps(estimator, pcs, demo_grasps)

    disconnect()


if __name__ == "__main__":
    main()
