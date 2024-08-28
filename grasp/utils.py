import time

import numpy as np

import owt.pb_utils as pbu

GRASP_MODES = ["graspnet", "gpd"]

#######################################################

GPD_GRIPPER_ADJUSTMENT = pbu.Pose(point=pbu.Point(x=-0.08))
GPD_TOOL_ADJUSTMENT = pbu.Pose(point=pbu.Point(x=0.035))


def local_gpd(points):
    from grasp.gpd_interface import generate_grasps

    entries = generate_grasps(points)
    grasps = [(grasp[:3], grasp[3:7]) for grasp in entries]
    scores = [grasp[7] for grasp in entries]
    return grasps, scores


def gpd_predict_grasps(points_world, camera_pose, use_tool=True):
    # Assumes camera_position = 0 0 0
    start_time = time.time()
    assert len(points_world) >= 1

    # reference_pose = Pose()
    reference_pose = pbu.Pose(pbu.point_from_pose(camera_pose))
    # reference_pose = camera_pose

    points_reference = pbu.tform_points(pbu.invert(reference_pose), points_world)
    grasps, scores = local_gpd(points_reference)
    grasps, scores = zip(
        *sorted(zip(grasps, scores), key=lambda pair: pair[-1], reverse=True)
    )

    print(
        "Grasps: {} | Min likelihood: {:.3f} | Max likelihood: {:.3f} | Time: {:.3f} sec".format(
            len(grasps),
            min(scores, default=-np.inf),
            max(scores, default=-np.inf),
            pbu.elapsed_time(start_time),
        )
    )

    adjustment = GPD_TOOL_ADJUSTMENT if use_tool else GPD_GRIPPER_ADJUSTMENT
    grasps = [
        pbu.multiply(reference_pose, grasp, adjustment) for grasp in grasps
    ]  # world_from_tool

    return grasps, scores


#######################################################

ADIAN_GRASPNET_ADJUSTMENT = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5))


def local_graspnet(points):
    from grasp.graspnet_interface import generate_grasps

    tforms, scores = generate_grasps(points, pc_colors=None)
    grasps = list(map(pbu.pose_from_tform, tforms))
    return grasps, scores


def graspnet_predict_grasps(points_world, camera_pose):
    start_time = time.time()
    assert len(points_world) >= 1

    # reference_pose = Pose()
    # reference_pose = Pose(point_from_pose(camera_pose))
    reference_pose = camera_pose

    points_reference = pbu.tform_points(pbu.invert(reference_pose), points_world)
    grasps, scores = local_graspnet(points_reference)
    grasps, scores = zip(
        *sorted(zip(grasps, scores), key=lambda pair: pair[-1], reverse=True)
    )
    grasps, scores = zip(*filter_identical_grasps(zip(grasps, scores)))

    print(
        "Grasps: {} | Min likelihood: {:.3f} | Max likelihood: {:.3f} | Time: {:.3f} sec".format(
            len(grasps),
            min(scores, default=-np.inf),
            max(scores, default=-np.inf),
            pbu.elapsed_time(start_time),
        )
    )

    adjustment = pbu.multiply(pbu.invert(GRASPNET_POSE))  # Pose(point=Point(x=-0.08)))
    # adjustment = ADIAN_GRASPNET_ADJUSTMENT
    grasps = [pbu.multiply(reference_pose, grasp, adjustment) for grasp in grasps]

    return grasps, scores
