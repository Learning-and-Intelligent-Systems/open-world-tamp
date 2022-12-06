import time

from pybullet_tools.utils import (
    INF,
    Point,
    Pose,
    elapsed_time,
    invert,
    multiply,
    point_from_pose,
    pose_from_tform,
    tform_points,
)

from open_world.planning.graspnet import GRASPNET_POSE, filter_identical_grasps
from open_world.simulation.lis import USING_ROS

GRASP_MODES = ["graspnet", "gpd"]


def query_grasp_server(points, grasp_mode="gpd", num_iterations=1):  # TODO: num_grasps
    import rospy
    from open_world_server.srv import Grasps

    from open_world.real_world.ros_utils import convert_ros_pose, create_cloud_msg

    assert grasp_mode in GRASP_MODES
    service_name = "/server/grasps"
    print("Waiting for service:", service_name)
    rospy.wait_for_service(service_name)
    grasps_service = rospy.ServiceProxy(service_name, Grasps)

    cloud_msg = create_cloud_msg(points)
    grasps = []
    scores = []
    for _ in range(num_iterations):
        grasps_response = grasps_service(cloud_msg, grasp_mode)
        grasps.extend(map(convert_ros_pose, grasps_response.grasps.poses))
        scores.extend(grasps_response.scores)
    grasps_service.close()

    return grasps, scores


#######################################################

GPD_GRIPPER_ADJUSTMENT = Pose(
    point=Point(x=-0.08)
)  # for l_gripper_palm_joint in pr2_l_gripper.urdf
GPD_TOOL_ADJUSTMENT = Pose(
    point=Point(x=0.035)
)  # (0.03376995027065277, -0.0005300119519233704, 0.0011900067329406738)


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
    reference_pose = Pose(point_from_pose(camera_pose))
    # reference_pose = camera_pose

    points_reference = tform_points(invert(reference_pose), points_world)
    if USING_ROS:
        grasps, scores = query_grasp_server(points_reference, grasp_mode="gpd")
    else:
        grasps, scores = local_gpd(points_reference)
    grasps, scores = zip(
        *sorted(zip(grasps, scores), key=lambda pair: pair[-1], reverse=True)
    )

    print(
        "Grasps: {} | Min likelihood: {:.3f} | Max likelihood: {:.3f} | Time: {:.3f} sec".format(
            len(grasps),
            min(scores, default=-INF),
            max(scores, default=-INF),
            elapsed_time(start_time),
        )
    )

    adjustment = GPD_TOOL_ADJUSTMENT if use_tool else GPD_GRIPPER_ADJUSTMENT
    grasps = [
        multiply(reference_pose, grasp, adjustment) for grasp in grasps
    ]  # world_from_tool

    return grasps, scores


#######################################################

ADIAN_GRASPNET_ADJUSTMENT = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5))


def local_graspnet(points):
    from grasp.graspnet_interface import generate_grasps

    tforms, scores = generate_grasps(points, pc_colors=None)
    grasps = list(map(pose_from_tform, tforms))
    return grasps, scores


def graspnet_predict_grasps(points_world, camera_pose):
    start_time = time.time()
    assert len(points_world) >= 1

    # reference_pose = Pose()
    # reference_pose = Pose(point_from_pose(camera_pose))
    reference_pose = camera_pose

    points_reference = tform_points(invert(reference_pose), points_world)
    if USING_ROS:
        grasps, scores = query_grasp_server(points_reference, grasp_mode="graspnet")
    else:
        grasps, scores = local_graspnet(points_reference)
    grasps, scores = zip(
        *sorted(zip(grasps, scores), key=lambda pair: pair[-1], reverse=True)
    )
    grasps, scores = zip(*filter_identical_grasps(zip(grasps, scores)))

    print(
        "Grasps: {} | Min likelihood: {:.3f} | Max likelihood: {:.3f} | Time: {:.3f} sec".format(
            len(grasps),
            min(scores, default=-INF),
            max(scores, default=-INF),
            elapsed_time(start_time),
        )
    )

    adjustment = multiply(invert(GRASPNET_POSE))  # Pose(point=Point(x=-0.08)))
    # adjustment = ADIAN_GRASPNET_ADJUSTMENT
    grasps = [
        multiply(reference_pose, grasp, adjustment) for grasp in grasps
    ]  # world_from_tool

    return grasps, scores
