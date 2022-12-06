#!/usr/bin/env python3

from __future__ import print_function

import ctypes
import datetime
import os
import struct
import sys
import time
import warnings

import numpy as np
import sklearn  # TODO: needs to be first likely due to PYTHONPATH naming conflicts
import torch  # TODO: needs to be first likely due to PYTHONPATH naming conflicts

warnings.filterwarnings("ignore")  # , category=DeprecationWarning)

from itertools import product

# from copy import copy

# scp segment_server.py demo@$ARIADNE:/home/demo/catkin_wsd/src/open-world-tamp

sys.path.extend(
    [
        #'pddlstream',
        "pybullet-planning",
        #'/usr/lib/python2.7/dist-packages', # for rospkg
    ]
)

# https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/tensorflow_filter.py
# https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/tensorflow_test.py

# TODO: all ROS should be the last import otherwise segfaults
import rospy

# from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from pybullet_tools.utils import AABB, RGB, SEPARATOR, CameraImage, elapsed_time
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2  # PointCloud
from sensor_msgs.point_cloud2 import (
    PointField,
    create_cloud,
    create_cloud_xyz32,
    read_points,
)

# from shape_msgs.msg import Mesh, Plane # http://wiki.ros.org/shape_msgs?distro=indigo
from depth_filter import DEPTH_TOPIC, FILTERED_TOPIC
from open_world.estimation.dnn import str_from_int_seg_general
from open_world.estimation.observation import (
    LabeledPoint,
    image_from_labeled,
    save_camera_images,
)
from open_world.simulation.entities import get_label_counts
from open_world.simulation.lis import CAMERA_OPTICAL_FRAME
from run_estimator import cloud_from_depth, create_parser, init_seg

# from tf import TransformListener, TransformerROS # TODO: ImportError: dynamic module does not define module export function (PyInit__tf2)
# from std_msgs.msg import Header, ColorRGBA
# from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
# from object_recognition_clusters.convert_functions import pose_to_mat, mat_to_pose, transform_point_cloud

DATE_FORMAT = "%y-%m-%d_%H-%M-%S"

CAMERA_FRAME = CAMERA_OPTICAL_FRAME  # head_mount_kinect_rgb_optical_frame
WORLD_FRAME = "odom_combined"
# POINT_CLOUD = "/passthrough2/points"

RGB_ENCODING = "rgb8"  # rgb8 | bgr8

ROS_NAME, _ = os.path.splitext(os.path.basename(__file__))

USE_FILTERED = False
if USE_FILTERED:
    DEPTH_TOPIC = FILTERED_TOPIC
# else:
#    DEPTH_TOPIC = '/head_mount_kinect/depth_registered/image'

##################################################


def get_date():
    return datetime.datetime.now().strftime(DATE_FORMAT)


# TODO: refactor to ros_utils.py


def read_rgb_image(bridge, data):
    # http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
    # http://docs.ros.org/en/jade/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html
    # http://docs.ros.org/en/indigo/api/cv_bridge/html/python/index.html
    # try:
    rgb_image = bridge.imgmsg_to_cv2(data, desired_encoding=RGB_ENCODING)
    # except CvBridgeError as e:
    # print(e)
    # height, width, channels = rgb_image.shape
    # print(start_time, data.header, height, width, channels)
    return np.array(rgb_image)


def read_depth_image(bridge, data):
    # try:
    depth_image = bridge.imgmsg_to_cv2(
        data, desired_encoding="passthrough"
    )  # passthrough
    # except CvBridgeError as e:
    # print(e)
    return np.array(depth_image)


def convert_image(bridge, image):
    # https://numpy.org/devdocs/reference/arrays.scalars.html#numpy.int8
    image = np.array(image).astype(dtype=np.uint8)
    # try:
    data = bridge.cv2_to_imgmsg(image, encoding=RGB_ENCODING)
    # data = copy(other_data)
    # data.data = data.data
    # TODO: rospy.exceptions.ROSSerializationException: field step must be unsigned integer type
    data.step = int(data.step)
    # dump_image(data)
    # except CvBridgeError as e:
    #    print(e)
    return data


def dump_image(image):
    print(image.header, image.height, image.width, image.encoding, image.step)


##################################################


def read_rgb(data):
    # https://answers.ros.org/question/208834/read-colours-from-a-pointcloud2-python/
    # cast float32 to int so that bitwise operations are possible
    s = struct.pack(">f", data)
    i = struct.unpack(">l", s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = pack & 0x000000FF
    # r,g,b values in the 0-255 range
    return RGB(r, g, b)


def read_point(fields, data, label=None):
    # TODO: parse instead of read
    assert fields is not None
    assert len(fields) == len(data)
    try:
        point = [data[fields.index(field)] for field in ["x", "y", "z"]]
    except ValueError:
        point = None
    try:
        rgb = read_rgb(data[fields.index("rgb")])
    except ValueError:
        rgb = None
    return LabeledPoint(point, rgb, label)


def iterate_box(box):
    return product(*(range(l, u + 1) for l, u in zip(*box)))


def iterate_rectangle(rectangle):
    (c1, r1), (c2, r2) = rectangle
    return [(c, r) for r in range(r1, r2 + 1) for c in range(c1, c1 + 1)]


def read_unordered_cloud(data, skip_nans=True, indices=None):
    # http://docs.ros.org/en/indigo/api/sensor_msgs/html/msg/PointCloud2.html
    # http://docs.ros.org/en/indigo/api/sensor_msgs/html/namespacesensor__msgs_1_1point__cloud2.html
    # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/tensorflow_filter.py#L212

    field_names = [field.name for field in data.fields]
    print(
        data.header.frame_id,
        field_names,
        data.width,
        data.height,
        data.width * data.height,
    )
    # field_names = ('x', 'y', 'z', 'rgb')
    # field_names = None # all fields
    return [
        read_point(field_names, p)
        for p in read_points(
            data, field_names=field_names, skip_nans=skip_nans, uvs=indices
        )
    ]


def read_ordered_cloud(data, box=None):
    if box is None:
        lower = (0, 0)
        upper = (data.width - 1, data.height - 1)  # column x row
        box = AABB(lower, upper)

    # If specified, then only return the points at the given coordinates. [default: empty list]
    # indices = [] # all points
    # indices = None # all points
    indices = iterate_box(box)  # order matters
    # indices = iterate_rectangle(box)
    dimensions = [(u - l + 1) for l, u in zip(*box)]

    points = read_unordered_cloud(data, skip_nans=False, indices=indices)
    return np.reshape(points, dimensions + [3])


def transform_tf(tf_listener, data):
    # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/tensorflow_test.py#L204
    # http://wiki.ros.org/tf/Tutorials/tf%20and%20Time%20%28Python%29
    # http://wiki.ros.org/tf/TfUsingPython
    try:
        # head_from_base = np.linalg.inv(self.tf_listener.asMatrix(HEAD_FRAME,\
        #    Header(0,rospy.Time(0), BASE_FRAME)))
        # head_from_base = np.linalg.inv(self.tf_listener.asMatrix(HEAD_FRAME,\
        #    Header(0,rospy.Time.now(), BASE_FRAME))) # rospy.get_rostime()
        world_from_kinect = tf_listener.asMatrix(WORLD_FRAME, data.header)
    except:
        world_from_kinect = None
        print("No transform avaliable!")


##################################################


class SegmentServer(object):
    def __init__(self, args):
        self.init_time = time.time()  # TODO: rospy.Time
        self.args = args
        # seg_network = None
        # if args.segmentation:
        # TODO: seg_from_args
        self.seg_network = init_seg(
            branch=self.args.segmentation_model,
            maskrcnn_rgbd=args.maskrcnn_rgbd,
            post_classifier=args.fasterrcnn_detection,
        )
        # seg_network, sc_network = init_networks(args)

        self.last_rgb = self.last_depth = self.last_cloud = None
        self.update_lock = self.segment_lock = False  # threading.Lock
        # self.history = []
        self.num_rgb = self.num_depth = self.num_segments = 0

        # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/tensorflow_test.py#L308
        # self.tf_listener = TransformListener()
        # self.transformer = TransformerROS()
        # http://mirror.umd.edu/roswiki/doc/diamondback/api/tf/html/python/tf_python.html
        # self.transformer.transformPointCloud(WORLD_FRAME, self.cloud) # Only works for PointCloud

        #########################

        # http://wiki.ros.org/robot_pose_publisher
        # robot_pose_publisher | robot_state_publisher
        self.last_state = None
        self.last_positions = {}
        self.joint_sub = rospy.Subscriber(
            "joint_states", JointState, self.joint_callback
        )

        #########################

        # rostopic: rosout
        # find . -name "pr2*.urdf"
        self.urdf = rospy.get_param("/robot_description")
        # write('robot_description.urdf', self.urdf)
        self.name = rospy.get_param("/robot/name")  # pr2mm
        self.type = rospy.get_param("/robot/type")  # pr2
        self.distro = rospy.get_param("/rosdistro")  # indigo
        self.version = rospy.get_param("/rosversion")  # 1.11.16

        self.camera_info = None
        self.cam_model = PinholeCameraModel()
        self.sub_kinect = rospy.Subscriber(
            "/head_mount_kinect/rgb/camera_info",
            CameraInfo,
            self.calibration_callback,
            queue_size=1,
        )  # depth | ir
        self.min_range = rospy.get_param(
            "/head_mount_kinect/disparity_depth/min_range"
        )  # , None)
        self.max_range = rospy.get_param(
            "/head_mount_kinect/disparity_depth/max_range"
        )  # rospy.has_param
        print("Depth range: [{:.3f}, {:.3f}]".format(self.min_range, self.max_range))
        self.bridge = CvBridge()

        #########################

        topic_template = "/{}/{{}}".format(ROS_NAME)
        # topic_template = '~{}'
        self.seg_publisher = rospy.Publisher(
            topic_template.format("segmented"), Image, queue_size=1
        )
        self.seg_color_publisher = rospy.Publisher(
            topic_template.format("segmented_color"), Image, queue_size=1
        )
        self.seg_points_publisher = rospy.Publisher(
            topic_template.format("segmented_points"), PointCloud2, queue_size=1
        )
        # self.markers_publisher = rospy.Publisher(topic_template.format('markers'), MarkerArray, queue_size=1)

        # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/base_navigation.py#L238
        # self.pose_est_sub = rospy.Subscriber('/robot_pose_ekf/odom_combined', PoseWithCovarianceStamped, self.dump_callback, queue_size=1)
        # self.pose_est_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.dump_callback, queue_size=1) # 2D Pose Estimate
        # self.nav_goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.dump_callback, queue_size=1) # 2D Nav Goal
        self.clicked_point_sub = rospy.Subscriber(
            "/clicked_point", PointStamped, self.dump_callback, queue_size=1
        )  # PublishedPoint
        # self.occupancy_grid_sub = rospy.Subscriber("/projected_map", OccupancyGrid, self.dump_callback, queue_size=1)
        # self.grid_pub = rospy.Publisher("~grid", OccupancyGrid, queue_size=1)
        # self.service = rospy.Service('~detect', Detect, self.handle_service)
        # http://wiki.ros.org/shape_msgs?distro=noetic

        # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/teleop_base_head.py#L143
        # from std_srvs.srv import Empty
        # rospy.wait_for_service('image_saver/save')
        # save = rospy.ServiceProxy('image_saver/save', Empty)
        # rosrun image_view image_saver image:=/head_mount_kinect/rgb/image_rect_color _save_all_image:=false __name:=image_saver _filename_format:=images/%04d.png

        #########################

        self.rgb_subscriber = rospy.Subscriber(
            "/head_mount_kinect/rgb/image_rect_color",
            Image,
            self.rgb_callback,
            queue_size=1,
        )
        # https://github.mit.edu/search?q=user%3Acaelan+depth_registered&type=Code
        # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/tensorflow_test.py
        # /head_mount_kinect/depth_registered/hw_registered/image_rect
        # /head_mount_kinect/depth_registered/hw_registered/image_rect_raw
        # /head_mount_kinect/depth_registered/image_raw
        # /head_mount_kinect/depth_registered/image
        self.depth_subscriber = rospy.Subscriber(
            DEPTH_TOPIC, Image, self.depth_callback, queue_size=1
        )
        self.cloud_subscriber = rospy.Subscriber(
            "/head_mount_kinect/depth_registered/points",
            PointCloud2,
            self.cloud_callback,
            queue_size=1,
        )
        # http://wiki.ros.org/openni_launch
        # http://wiki.ros.org/rgbd_launch
        # http://wiki.ros.org/image_proc
        # http://wiki.ros.org/depth_image_proc
        # http://wiki.ros.org/pcl_ros/Tutorials/filters
        # http://wiki.ros.org/opencv_apps

        # TODO: read_points versus ApproximateTimeSynchronizer

        #########################

        # from open_world_server.srv import Grasps
        # service_name = '/server/grasps'
        # rospy.wait_for_service(service_name)
        # self.grasps_service = rospy.ServiceProxy(service_name, Grasps)

    #########################

    def dump_callback(self, data):
        # TODO: text or GUI to enter attribute annotation
        print(data)

    def calibration_callback(self, data):
        # http://docs.ros.org/en/indigo/api/image_geometry/html/python/index.html
        # http://docs.ros.org/en/indigo/api/sensor_msgs/html/msg/CameraInfo.html
        # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/044d5b22f6976f7d1c7ba53cc445ce01a557646e/perception_tools/ros_perception.py#L91
        # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/tensorflow_test.py#L129
        if self.camera_info is not None:
            return self.camera_info
        self.camera_info = data
        self.cam_model.fromCameraInfo(self.camera_info)
        # assert CAMERA_OPTICAL_FRAME != self.camera_info.header.frame_id # head_mount_kinect_rgb_optical_frame

        # print(self.camera_info)
        # print(self.cam_model.intrinsicMatrix()) # 3 x 3 camera_matrix
        # print(self.cam_model.projectionMatrix()) # 3 x 4 camera_matrix
        # print(self.cam_model.rotationMatrix()) # 3 x 3 identity matrix

        # self.cam_model.projectPixelTo3dRay(pixel)
        # self.cam_model.project3dToPixel(point)
        # points = [world_from_kinect.dot(np.concatenate([
        #    distance * np.array(self.cam_model.projectPixelTo3dRay(pixel)), [1]]))[:3] for pixel in pixels]

        return self.camera_info

    @property
    def camera_matrix(self):
        # return CAMERA_MATRIX
        return np.array(self.cam_model.intrinsicMatrix())

    def joint_callback(self, data):
        if self.update_lock:
            return False
        self.last_state = data
        self.last_positions = dict(zip(data.name, data.position))
        return True

    def rgb_callback(self, data):
        # NOTE: do not call segment in a callback because it slows new messages
        if self.update_lock:
            return False
        self.last_rgb = data
        self.num_rgb += 1
        return True

    def depth_callback(self, data):
        if self.update_lock:
            return False
        self.last_depth = data
        self.num_depth += 1
        return True

    def cloud_callback(self, data):
        if self.update_lock:
            return False
        self.last_cloud = data
        # try:
        #     grasps_response = self.grasps_service(data)
        #     data = grasps_response.grasps
        #     grasps = data.poses
        #     print(data.header, grasps)
        # except rospy.ServiceException as e:
        #     print("Service call failed: {}".format(e))
        return True

    def publish_cloud(self, point_cloud):
        # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/tensorflow_test.py#L342
        cloud_data = create_cloud_xyz32(self.last_cloud.header, point_cloud)
        # http://docs.ros.org/en/indigo/api/sensor_msgs/html/point__cloud2_8py_source.html
        field = PointField(
            "x", 0, PointField.FLOAT32, 1
        )  # TODO: could publish label here if needed
        field = PointField("y", 4, PointField.FLOAT32, 1)
        cloud_data = create_cloud(
            self.last_cloud.header, self.last_cloud.header.fields, point_cloud
        )
        self.seg_points_publisher.publish(cloud_data)
        # TODO: TypeError: Invalid number of arguments, args should be
        #  ['header', 'height', 'width', 'fields', 'is_bigendian', 'point_step', 'row_step', 'data', 'is_dense'] args

    #########################

    def segment(self):
        # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_perception/blob/master/perception_tools/tensorflow_detector.py
        start_time = time.time()
        if self.segment_lock or any(
            data is None
            for data in [
                self.camera_info,
                self.last_rgb,
                self.last_depth,
                self.last_positions,
            ]
        ):  # , self.last_cloud]):
            return None
        self.update_lock = self.segment_lock = True
        self.num_segments += 1

        positions = dict(self.last_positions)  # TODO: use self.last_positions
        rgb = read_rgb_image(self.bridge, self.last_rgb)
        depth = read_depth_image(self.bridge, self.last_depth)
        height, width, channels = rgb.shape
        print(
            "{} | Width: {} | Height: {} | Time: {:.3f}".format(
                self.num_segments, width, height, elapsed_time(self.init_time)
            )
        )
        # print(self.last_rgb.header.frame_id, self.last_depth.header.frame_id) # head_mount_kinect_rgb_optical_frame

        print(
            "RGB: {} | Depth: {} | Segment: {} | Time: {:.3f}".format(
                self.num_rgb,
                self.num_depth,
                self.num_segments,
                elapsed_time(self.init_time),
            )
        )

        # points = read_ordered_cloud(self.last_cloud)
        # print('Points:', len(points))
        self.update_lock = False

        depth_nanless = depth.copy()
        nan_indices = np.isnan(depth_nanless)
        depth_nanless[nan_indices] = 0  # self.max_range
        print(
            "Depth range: [{:.3f}, {:.3f}]".format(
                np.min(depth_nanless), np.max(depth_nanless)
            )
        )

        # TODO fill depth. https://github.mit.edu/Learning-and-Intelligent-Systems/open-world-tamp/blob/master/test_observations.py#L80
        point_cloud = cloud_from_depth(
            self.camera_matrix, depth_nanless, max_depth=self.max_range
        )
        int_seg = self.seg_network.get_seg(
            rgb, point_cloud=point_cloud, return_int=True, depth_image=depth_nanless
        )
        # int_seg[nan_indices, :] = 0

        output_seg = np.zeros(int_seg.shape[:2] + (3,), dtype=np.uint8)
        output_seg[..., : int_seg.shape[2]] = int_seg
        self.seg_publisher.publish(
            convert_image(self.bridge, output_seg)
        )  # TODO: 2 channel encoding

        str_seg = str_from_int_seg_general(
            int_seg, use_classifer=self.args.fasterrcnn_detection
        )
        print("Segmentation:", get_label_counts(str_seg))

        color_seg = image_from_labeled(str_seg)
        # if self.args.save:
        #    save_image(os.path.join('.', 'segmented.png'), color_seg)  # [0, 255]
        self.seg_color_publisher.publish(convert_image(self.bridge, color_seg))

        # TODO: publish segmented point cloud
        # self.seg_points_publisher.publish(self.last_cloud)
        # self.publish_cloud(point_cloud)

        depth = depth_nanless
        camera_image = CameraImage(
            rgb, depth, str_seg, positions, self.camera_matrix
        )  # TODO: camera_pose
        if self.args.save:
            save_camera_images(camera_image)

        self.segment_lock = False
        print("Time: {:.3f}".format(elapsed_time(start_time)))
        return camera_image

    def wait_until_segment(self, **kwargs):
        while True:
            camera_image = self.segment(**kwargs)
            if camera_image is not None:
                return camera_image


##################################################


def main():
    np.set_printoptions(
        precision=3, threshold=3, suppress=True
    )  # , edgeitems=1) #, linewidth=1000)
    parser = create_parser()
    args = parser.parse_args()
    # TODO: publish more of the perception system

    # rospy.get_param('~model_dir'),
    rospy.init_node(ROS_NAME)

    start_time = time.time()
    server = SegmentServer(args)
    # images = []
    while not rospy.is_shutdown():
        print(SEPARATOR)
        camera_image = server.wait_until_segment()
        # images.append(camera_image)
        # print('{:.3f}: saved {} observations to {}'.format(
        #    elapsed_time(start_time), len(observations), filename))
        # TODO: rospy.sleep
    # rospy.spin()


if __name__ == "__main__":
    main()
