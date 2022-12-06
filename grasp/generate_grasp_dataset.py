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
sys.path.extend([
    'pddlstream',
    'pybullet-planning',
    #'pddlstream/examples/pybullet/utils',
])

import copy
import math
import random
from collections import namedtuple

from pybullet_tools.pr2_problems import create_table
from pybullet_tools.pr2_utils import (DRAKE_PR2_URDF, PR2_CAMERA_MATRIX,
                                      PR2_GROUPS, REST_LEFT_ARM,
                                      attach_viewcone, get_pr2_field_of_view,
                                      get_viewcone, pixel_from_ray,
                                      ray_from_pixel, rightarm_from_leftarm)
from pybullet_tools.utils import (BLACK, GREY, PI, RED, TAN, TEMP_DIR, WHITE,
                                  HideOutput, Pose, aabb_from_points,
                                  add_data_path, add_line, apply_alpha,
                                  connect, create_mesh, create_obj,
                                  create_plane, disconnect, draw_aabb,
                                  draw_mesh, draw_point, draw_pose, get_aabb,
                                  get_aabb_center, get_camera, get_image,
                                  get_link_pose, image_from_segmented, invert,
                                  joints_from_names, link_from_name,
                                  load_model, load_pybullet, mesh_from_points,
                                  multiply, point_from_pose, pose_from_tform,
                                  save_image, set_camera, set_camera_pose,
                                  set_camera_pose2, set_joint_positions,
                                  set_point, set_pose, stable_z, tform_point,
                                  wait_if_gui)
from pybullet_tools.voxels import MAX_PIXEL_VALUE

from grasp.graspnet_interface import visualize_grasps

np.set_printoptions(precision=3, threshold=3, edgeitems=1, suppress=True) #, linewidth=1000)

MODEL_PATH = './models'
LTAMP_PR2 = os.path.join(MODEL_PATH, 'pr2_description/pr2.urdf')
object_paths = [
    'ycb/003_cracker_box/textured.obj',
    'ycb/004_sugar_box/textured.obj',
    'ycb/005_tomato_soup_can/textured.obj',
    'ycb/006_mustard_bottle/textured.obj',
    'ycb/008_pudding_box/textured.obj',
    'ycb/009_gelatin_box/textured.obj',
    'ycb/010_potted_meat_can/textured.obj',
    'ycb/024_bowl/textured.obj',
]
object_angles = [
    0, 
    math.pi/4,
    math.pi/2,
    -math.pi/4,
    -math.pi/2,
    math.pi
]



#from examples.test_visiblity import *
#from learn_tools.collectors.collect_stir import *


# https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pointCloudFromCameraImage.py
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getCameraImageTest.py
# https://www.geeksforgeeks.org/python-pil-image-save-method/

#######################################################

LIS_PR2 = True
if LIS_PR2:
    CAMERA_FRAME = 'head_mount_kinect_rgb_link'
    CAMERA_OPTICAL_FRAME = 'head_mount_kinect_rgb_optical_frame'
    WIDTH, HEIGHT = 640, 480
    FX, FY, CX, CY = 525., 525., 319.5, 239.5
    #INTRINSICS = get_camera_matrix(WIDTH, HEIGHT, FX, FY)
    INTRINSICS = np.array([[FX, 0., CX],
                           [0., FY, CY],
                           [0., 0., 1.]])
else:
    CAMERA_FRAME = 'high_def_frame'
    CAMERA_OPTICAL_FRAME = 'high_def_optical_frame' # HEAD_LINK_NAME
    INTRINSICS = PR2_CAMERA_MATRIX
    # TODO: WIDTH, HEIGHT

#######################################################

LabeledPoint = namedtuple('LabledPoint', ['point', 'color', 'body'])

def set_group_positions(pr2, group_positions):
    for group, positions in group_positions.items():
        joints = joints_from_names(pr2, PR2_GROUPS[group])
        assert len(joints) == len(positions)
        set_joint_positions(pr2, joints, positions)

#######################################################

def convexify(labeled_points, points_to_add = 10000):
    new_labeled_points = copy.deepcopy(labeled_points)
    for _ in range(points_to_add):
        pntidx = random.sample(labeled_points, 2)
        t = random.uniform(0, 1)
        new_point = tuple([pntidx[0].point[i]*t+pntidx[1].point[i]*(1-t) for i in range(3)])
        new_labeled_points.append(LabeledPoint(new_point, pntidx[0].color, pntidx[0].body))
    return new_labeled_points


def main(object_path, object_angle):
    OBJ_PATH = os.path.join(MODEL_PATH, object_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--viewer', action='store_true',
                        help='')
    args = parser.parse_args()

    connect(use_gui=args.viewer)
    draw_pose(Pose(), length=1)
    with HideOutput(): # enable=None):
        add_data_path()
        floor = create_plane(color=TAN)
        table = create_table(leg_color=GREY, cylinder=False)
        obj = create_obj(OBJ_PATH, color=WHITE) #, **kwargs)

        if LIS_PR2:
            pr2 = load_pybullet(LTAMP_PR2)
        else:
            pr2 = load_model(DRAKE_PR2_URDF)

    set_pose(obj, ([0, 0, stable_z(obj, table)], p.getQuaternionFromEuler([0, 0, object_angle])))

    group_positions = {
        'base': [-1., 0, 0],
        'left_arm': REST_LEFT_ARM,
        'right_arm': rightarm_from_leftarm(REST_LEFT_ARM),
        'torso': [0.2],
        'head': [0, PI/4],
    }
    set_group_positions(pr2, group_positions)

    table_aabb = get_aabb(table)
    draw_aabb(table_aabb)
    x, y, _ = get_aabb_center(table_aabb)

    camera_matrix = INTRINSICS
    camera_link = link_from_name(pr2, CAMERA_OPTICAL_FRAME)
    camera_pose = get_link_pose(pr2, camera_link)
    draw_pose(camera_pose, length=1) # TODO: attach to camera_link

    distance = 2.5
    target_point = tform_point(camera_pose, np.array([0, 0, distance]))
    attach_viewcone(pr2, depth=distance, head_name=CAMERA_FRAME, camera_matrix=camera_matrix, color=RED)

    camera_point = point_from_pose(camera_pose)
    _, vertical_fov = get_pr2_field_of_view(camera_matrix=camera_matrix)
    rgb_image, depth_image, seg_image = get_image(
        camera_pos=camera_point, target_pos=target_point,
        width=WIDTH, height=HEIGHT, vertical_fov=vertical_fov,
        tiny=False, segment=True, segment_links=False)

    save_image(os.path.join(TEMP_DIR, 'rgb.png'), rgb_image) # [0, 255]
    save_image(os.path.join(TEMP_DIR, 'depth.png'), depth_image) # [0, 1]
    if seg_image is not None:
        segmented_image = image_from_segmented(seg_image, color_from_body=None)
        save_image(os.path.join(TEMP_DIR, 'segmented.png'), segmented_image) # [0, 1]

    step_size = 1
    labeled_points = []
    for r in range(0, depth_image.shape[0], step_size):
        for c in range(0, depth_image.shape[1], step_size):
            body, link = seg_image[r, c, :]
            if body not in [obj]:
                continue
            pixel = [c, r]  # NOTE: width, height
            #print(pixel, pixel_from_ray(camera_matrix, ray_from_pixel(camera_matrix, pixel)))
            ray = ray_from_pixel(camera_matrix, pixel)
            depth = depth_image[r, c]
            point_camera = depth*ray
            point_world = tform_point(multiply(camera_pose), point_camera)
            point_world = [point_world[1], point_world[2],  point_world[0]]
            color = rgb_image[r, c, :] / MAX_PIXEL_VALUE
            labeled_points.append(LabeledPoint(point_world, color, body))
            #draw_point(point_world, size=0.01, color=color) # TODO: adjust size based on step_size
            #add_line(camera_point, point_world, color=color)

    points = [point.point for point in labeled_points]
    mean_color = np.average([point.color for point in labeled_points], axis=0) # average | median
    centroid = np.average(points, axis=0)
    obj_aabb = aabb_from_points(points)
    origin_point = centroid
    origin_point[2] = obj_aabb[0][2]
    origin_pose = Pose(origin_point) # PCA
    draw_pose(origin_pose)

    #draw_aabb(obj_aabb, color=mean_color)
    obj_mesh = mesh_from_points(points)
    #draw_mesh(obj_mesh, color=mean_color)
    obj_approx = create_mesh(obj_mesh, under=True, color=mean_color)
    # TODO: choose origin
    set_pose(obj, Pose())
    # TODO: simplify meshes

    # plot the full 3d point cloud in mayavi
    pc = np.array([p.point for p in labeled_points])
    pc_color = np.array([p.color for p in labeled_points])[:,:3]*255

    # Pass through the graspnet for inference
    visualize_grasps(pc, pc_color, object_path.split("/")[-2]+"_"str(object_angle))
    disconnect()

if __name__ == '__main__':
    for object_path in object_paths:
        for angle in object_angles:
            main(object_path, object_angle)
