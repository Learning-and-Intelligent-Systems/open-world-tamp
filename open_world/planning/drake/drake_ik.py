#!/usr/bin/env python3
from __future__ import print_function

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
    ]
)
from pybullet_tools.utils import (
    CameraImage,
    invert,
    pixel_from_point,
    tform_point,
)

from open_world.estimation.observation import (
    save_camera_images,
)

from open_world.simulation.policy import (
    CameraImage,
    link_seg_from_gt,
)

SAVE_DIR = "temp_graphs/"

def get_camera_image(args, robot):
    [camera] = robot.cameras
    camera_image = camera.get_image(segment_links=True)
    rgb, depth, predicted_seg, camera_pose, camera_matrix = (
        camera_image.rgbPixels,
        camera_image.depthPixels,
        camera_image.segmentationMaskBuffer,
        camera_image.camera_pose,
        camera_image.camera_matrix,
    )
    camera_image = CameraImage(
        rgb, depth, link_seg_from_gt(predicted_seg), camera_pose, camera_matrix
    )
    save_camera_images(camera_image, directory=SAVE_DIR)

    return camera_image


def update_visibility(camera_image, visibility_grid, **kwargs):
    surface_aabb = visibility_grid.aabb
    camera_pose, camera_matrix = camera_image[-2:]
    grid = visibility_grid
    for voxel in grid.voxels_from_aabb(surface_aabb):
        center_world = grid.to_world(grid.center_from_voxel(voxel))
        center_camera = tform_point(invert(camera_pose), center_world)
        distance = center_camera[2]
        pixel = pixel_from_point(camera_matrix, center_camera)
        if pixel is not None:
            r, c = pixel
            depth = camera_image.depthPixels[r, c]
            if distance <= depth:
                grid.set_free(voxel)
    return grid


def reset_robot(robot):
    conf = robot.get_default_conf()
    for group, positions in conf.items():
        robot.set_group_positions(group, positions)
