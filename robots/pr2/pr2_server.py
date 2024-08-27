import functools
import threading
import zlib

import actionlib
import pickle5
import rospy  # TODO: causes FastDownward segfault
import zmq
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist, Vector3
from moveit_msgs.msg import DisplayRobotState  # , CollisionObject
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from object_recognition_msgs.msg import TableArray
from open_world.real_world.ros_utils import (create_mesh_markers,
                                             create_table_msgs)
from pr2_controllers_msgs.msg import (JointTrajectoryAction,
                                      JointTrajectoryGoal,
                                      Pr2GripperCommandAction,
                                      Pr2GripperCommandGoal,
                                      SingleJointPositionAction,
                                      SingleJointPositionGoal)
from pr2_gripper_sensor_msgs.msg import (PR2GripperEventDetectorAction,
                                         PR2GripperEventDetectorData,
                                         PR2GripperEventDetectorGoal,
                                         PR2GripperGrabAction,
                                         PR2GripperGrabGoal,
                                         PR2GripperReleaseAction,
                                         PR2GripperReleaseGoal)
from segment_server import CAMERA_FRAME, SegmentServer, get_date
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import MarkerArray

# TODO: hide moveit_msgs import
# from soundplay_msgs.msg import SoundRequest

# from actionlib_msgs/GoalStatus
# http://docs.ros.org/melodic/api/actionlib_msgs/html/msg/GoalStatus.html
# PENDING = 0
# ACTIVE = 1
# PREEMPTED = 2
# SUCCEEDED = 3
# ABORTED = 4
# REJECTED = 5
# PREEMPTING = 6
# RECALLING = 7
# RECALLED = 8
# LOST = 9

# http://wiki.ros.org/pr2_controllers/Tutorials/Moving%20the%20arm%20using%20the%20Joint%20Trajectory%20Action
# http://wiki.ros.org/joint_trajectory_action
# TODO: constraints/goal_time
# /l_arm_controller/joint_trajectory_action_node/constraints/goal_time
# /opt/ros/indigo/share/pr2_controller_configuration/pr2_arm_controllers.yaml
# http://wiki.ros.org/robot_mechanism_controllers/JointTrajectoryActionController
# http://wiki.ros.org/pr2_controller_configuration
# vim /opt/ros/indigo/share/pr2_controller_configuration/pr2_arm_controllers.yaml
# rostopic echo /l_arm_controller/state
# http://docs.ros.org/api/actionlib/html/simple__action__client_8py_source.html
# http://docs.ros.org/api/actionlib/html/classactionlib_1_1simple__action__client_1_1SimpleActionClient.html
# The controller will interpolate between these points using cubic splines.
# rosservice info /l_arm_controller/query_state
# http://wiki.ros.org/pr2_controller_manager/safety_limits
# https://www.clearpathrobotics.com/wp-content/uploads/2014/08/pr2_manual_r321.pdf
# http://wiki.ros.org/robot_mechanism_controllers/JointSplineTrajectoryController
# rosnode info /realtime_loop


while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))

    print("Received request: {}".format(message))

    #  Send reply back to client
    globals()[message["message_name"]](message)
