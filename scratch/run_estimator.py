#!/usr/bin/env python3

from __future__ import print_function

import os

import numpy as np
from open_world.estimation.clustering import cluster_points
from open_world.estimation.dnn import DEFAULT_DEBUG, init_sc, init_seg
from open_world.estimation.geometry import (cloud_from_depth,
                                            estimate_surface_mesh)
from open_world.estimation.observation import (iterate_point_cloud,
                                               save_camera_images)
from open_world.simulation.entities import UNKNOWN, Object, get_label_counts
from open_world.simulation.environment import create_ycb
from open_world.simulation.lis import Z_EPSILON
from pybullet_tools.utils import (PI, TEMP_DIR, CameraImage, Euler, Point,
                                  Pose, ensure_dir, set_pose, stable_z)


def test_completion(
    sc_network,
    camera_image,
    known_objects,
    world_frame=True,
    concave=False,
    use_instance_label=False,
    min_points=15,
    **kwargs
):
    labeled_points = [
        labeled_point
        for labeled_point in iterate_point_cloud(camera_image)
        if (
            isinstance(labeled_point.label[1], Object)
            and (labeled_point.label[1] not in known_objects)
        )
        or (
            isinstance(labeled_point.label[1], str)
            and (labeled_point.label[1] != UNKNOWN)
        )
    ]
    clustered_points = cluster_points(
        labeled_points,
        use_instance_label=use_instance_label,
        min_points=min_points,
        **kwargs
    )

    for cluster in clustered_points:
        labeled_cluster = [labeled_points[index] for index in cluster]
        points = [labeled_point.point for labeled_point in labeled_cluster]
        min_z = np.min(points, axis=0)[2]
        surface_pose = Pose(Point(z=min_z))  # TODO: apply world_frame here
        body = estimate_surface_mesh(
            labeled_cluster,
            surface_pose,
            project_base=False,
            sc_network=sc_network,
            camera_image=camera_image,
            concave=concave,
        )
        # if concave: # move into estimate_surface_mesh
        #     body = decompose_body(body)

        # body = complete_shape(sc_network, points, color=color)
        if not world_frame:
            set_pose(body, camera_image.camera_pose)
        color = np.median(
            [labeled_point.color for labeled_point in labeled_cluster], axis=0
        )  # mean | median
        print(len(cluster), body, color)


################################################################################


def fuse_segmentation(predicted_seg, bullet_seg):
    fused_seg = np.array(predicted_seg, dtype=object)
    for r in range(fused_seg.shape[0]):
        for c in range(fused_seg.shape[1]):
            if (bullet_seg[r, c, 1] is None) or isinstance(
                bullet_seg[r, c, 1], Object
            ):  # is empty or is known object(table)
                fused_seg[r, c] = bullet_seg[r, c]
    return fused_seg


def fuse_predicted_labels(
    seg_network,
    camera_image,
    fuse=False,
    use_depth=False,
    debug=DEFAULT_DEBUG,
    **kwargs
):
    rgb, depth, bullet_seg, _, camera_matrix = camera_image
    if fuse:
        print("Ground truth:", get_label_counts(bullet_seg))

    point_cloud = None
    if use_depth:
        point_cloud = cloud_from_depth(camera_matrix, depth)

    if debug:
        predicted_seg = seg_network.get_seg(
            rgb[:, :, :3],
            point_cloud=point_cloud,
            depth_image=depth,
            return_int=False,
            debug=True,
            **kwargs
        )
        # save image here
        import matplotlib.pyplot as plt

        ensure_dir(TEMP_DIR)
        plt.savefig(os.path.join(TEMP_DIR, "merged_seg.png"))
        plt.close()
    else:
        predicted_seg = seg_network.get_seg(
            rgb[:, :, :3],
            point_cloud=point_cloud,
            depth_image=depth,
            return_int=False,
            **kwargs
        )

    # TODO clean up params
    print("Predictions:", get_label_counts(predicted_seg))

    if fuse:
        predicted_seg = fuse_segmentation(predicted_seg, bullet_seg)
    return CameraImage(rgb, depth, predicted_seg, *camera_image[3:])


def init_networks(args):
    sc_network = seg_network = None
    if args.shape_completion:
        sc_network = init_sc(branch=args.shape_completion_model)
    if args.segmentation:
        seg_network = init_seg(
            branch=args.segmentation_model,
            maskrcnn_rgbd=args.maskrcnn_rgbd,
            post_classifier=args.fasterrcnn_detection,
        )
    return seg_network, sc_network


def observe_and_segment(real_world, seg_network, args, **kwargs):
    # if args.observable:
    #     return real_world.movable
    robot = real_world.robot
    [camera] = robot.cameras
    camera_image = camera.get_image()  # TODO: remove_alpha
    camera_image = real_world.label_image(camera_image)  # Applies known labeling
    # real_world.disable()
    if args.segmentation:
        camera_image = fuse_predicted_labels(
            seg_network,
            camera_image,
            use_depth=args.segmentation_model != "maskrcnn",
            **kwargs
        )
    if args.save:
        save_camera_images(camera_image)
    return camera_image


################################################################################


def create_mustard(table):
    obj1 = create_ycb(
        "potted_meat_can"
    )  # cracker_box | tomato_soup_can | potted_meat_can | bowl
    set_pose(obj1, Pose(Point(z=stable_z(obj1, table) + Z_EPSILON), Euler(yaw=PI / 4)))

    obj2 = create_ycb("cracker_box")
    set_pose(
        obj2,
        Pose(Point(y=0.3, z=stable_z(obj2, table) + Z_EPSILON), Euler(yaw=-PI / 16)),
    )

    obj3 = create_ycb("mustard_bottle")
    set_pose(
        obj3, Pose(Point(y=-0.3, z=stable_z(obj3, table) + Z_EPSILON), Euler(yaw=0))
    )

    return [obj1, obj2, obj3]


def create_bowl_stack(table):
    obj1 = create_ycb("bowl")  # cracker_box | tomato_soup_can | potted_meat_can | bowl
    set_pose(obj1, Pose(Point(z=stable_z(obj1, table) + Z_EPSILON), Euler(yaw=PI / 4)))

    obj2 = create_ycb("cracker_box")
    set_pose(
        obj2,
        Pose(Point(z=stable_z(obj2, table) + Z_EPSILON + 0.05), Euler(yaw=-PI / 16)),
    )

    obj3 = create_ycb("bowl")
    set_pose(
        obj3, Pose(Point(y=-0.3, z=stable_z(obj3, table) + Z_EPSILON), Euler(yaw=0))
    )

    return [obj1, obj2, obj3]
