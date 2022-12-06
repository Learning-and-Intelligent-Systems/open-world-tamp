import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion
from object_recognition_msgs.msg import Table, TableArray
from pybullet_tools.utils import (
    BLACK,
    RED,
    apply_alpha,
    get_pose,
    point_from_pose,
    quat_from_pose,
    unit_pose,
)
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import ColorRGBA, Header

# from object_recognition_clusters import cluster_bounding_box_finder
from visualization_msgs.msg import Marker, MarkerArray

from open_world.planning.grasping import mesh_from_obj

BASE_FRAME = "base_footprint"  # base_link | base_footprint | odom_combined
FOREVER = 0
NAMESPACE = "default"  # TODO: update

# TODO: http://wiki.ros.org/rviz/Tutorials
# TODO: http://wiki.ros.org/rviz/DisplayTypes
# TODO: http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/InteractiveMarker.html
# http://docs.ros.org/en/indigo/api/visualization_msgs/html/msg/Marker.html

##################################################


def parse_triangle_msg(triangle):
    return tuple(triangle.vertex_indices)


def convert_ros_position(position):
    return [position.x, position.y, position.z]


def convert_ros_orientation(orientation):
    return [orientation.x, orientation.y, orientation.z, orientation.w]


def convert_ros_pose(pose):
    return convert_ros_position(pose.position), convert_ros_orientation(
        pose.orientation
    )


##################################################


def create_pose_msg(pose):
    return Pose(
        position=Point(*point_from_pose(pose)),
        orientation=Quaternion(*quat_from_pose(pose)),
    )


def create_rgba(color, **kwargs):
    return ColorRGBA(*apply_alpha(color, **kwargs))


def create_cloud_msg(points, frame=None):
    header = Header()
    if frame is not None:
        header.frame_id = frame
    return create_cloud_xyz32(header, points)


##################################################


def create_table_msg(surface, frame=BASE_FRAME):
    # https://github.com/caelan/SS-Isaac/blob/d58872173a04e4d6602358c22662ba31fe6c06c2/src/deepim.py#L15
    # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/move_group_python_interface_tutorial.py#L116
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/044d5b22f6976f7d1c7ba53cc445ce01a557646e/perception_tools/ros_perception.py#L25
    table_msg = Table()
    table_msg.header.frame_id = frame
    table_msg.pose = create_pose_msg(surface.pose)
    table_msg.convex_hull = [Point(*point) for point in surface.vertices]
    return table_msg


def create_table_msgs(surfaces, frame=BASE_FRAME):
    table_array_msg = TableArray()
    table_array_msg.header.frame_id = frame
    table_array_msg.tables = [create_table_msg(surface) for surface in surfaces]
    return table_array_msg


def create_bbox_marker(
    bbox, namespace=NAMESPACE, pose=unit_pose(), color=RED, index=0, duration=FOREVER
):
    # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/utils.py#L287
    # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/utils.py#L359
    (np.array(bbox[0]) + np.array(bbox[1])) / 2
    extents = np.array(bbox[1]) - np.array(bbox[0])
    m = Marker()
    m.header.frame_id = BASE_FRAME
    m.type = Marker.CUBE
    # from object_recognition_clusters.convert_functions import get_xyz, get_xyzw, pose_to_mat, mat_to_pose
    # m.pose = mat_to_pose(pose_to_mat(pose).dot(get_translation(center)))
    m.pose = create_pose_msg(pose)
    m.scale.x, m.scale.y, m.scale.z = extents
    m.ns = namespace
    m.lifetime = rospy.Duration(duration)
    m.id = index
    m.color = create_rgba(color, alpha=0.5)
    return m


def create_delete_marker(namespace=NAMESPACE):
    m = Marker()
    # m.header.frame_id = BASE_FRAME
    m.action = 3  # Marker.DELETEALL # DELETE # TODO: Segmentation fault (core dumped)
    # m.ns = namespace # Causes a seg fault
    return m


# def create_dummy_markers(namespace, start, end):
#     markers = []
#     for i in range(start, end):
#         m = Marker()
#         m.header.frame_id = BASE_FRAME
#         m.type = Marker.CUBE
#         m.scale.x, m.scale.y, m.scale.z = 1, 1, 1
#         m.ns = namespace
#         m.lifetime = rospy.Duration(FOREVER)
#         m.id = i
#         m.color = ColorRGBA(0, 0, 0, 0)
#         markers.append(m)
#     return markers


def create_text_marker(
    text,
    height=10,
    namespace=NAMESPACE,
    pose=unit_pose(),
    color=BLACK,
    index=0,
    duration=FOREVER,
):
    # TODO: https://github.com/RobotWebTools/ros3djs/issues/103
    m = Marker()
    m.header.frame_id = BASE_FRAME
    m.type = Marker.TEXT_VIEW_FACING
    m.text = str(text)
    m.scale.z = height
    m.pose = pose
    m.ns = namespace
    m.lifetime = rospy.Duration(duration)
    m.color = create_rgba(color, alpha=1.0)
    m.id = index
    return m


def create_mesh_marker(
    mesh, namespace=NAMESPACE, pose=unit_pose(), color=RED, index=0, duration=FOREVER
):
    verts, faces = mesh
    m = Marker()
    m.header.frame_id = BASE_FRAME
    m.type = Marker.TRIANGLE_LIST
    m.scale.x, m.scale.y, m.scale.z = 1, 1, 1
    m.points = [Point(*verts[i]) for face in faces for i in face]
    m.pose = create_pose_msg(pose)
    m.ns = namespace
    m.lifetime = rospy.Duration(duration)
    m.color = create_rgba(color, alpha=1.0)
    m.id = index
    return m


def create_points_marker(
    points,
    namespace=NAMESPACE,
    pose=unit_pose(),
    color=RED,
    scale=1e-2,
    index=0,
    duration=FOREVER,
):
    m = Marker()
    m.header.frame_id = BASE_FRAME
    m.type = Marker.POINTS
    m.scale.x, m.scale.y = scale, scale
    m.points = [Point(*p) for p in points]
    m.pose = pose
    m.ns = namespace
    m.lifetime = rospy.Duration(duration)
    m.color = create_rgba(color, alpha=1.0)
    m.id = index
    return m


def create_mesh_markers(objects):
    markers = [
        create_delete_marker(),
    ]
    for i, obj in enumerate(objects):
        mesh = mesh_from_obj(obj)
        namespace = obj.category  # name | category
        # color = RED
        color = obj.color
        pose = get_pose(obj)
        # print(namespace, i, obj)
        markers.extend(
            [
                create_mesh_marker(
                    mesh,
                    namespace=namespace,
                    pose=pose,
                    color=color,
                    index=i,
                    duration=FOREVER,
                ),
                # create_text_marker(obj.name, namespace=namespace, pose=pose, # TODO: AttributeError: 'tuple' object has no attribute 'position'
                #                   index=len(bodies) + i, duration=FOREVER),
            ]
        )

    # return markers
    return MarkerArray(markers)
