#!/usr/bin/env python2


from __future__ import print_function

import sys

sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
    ]
)
import os
import time
from collections import namedtuple

import numpy as np
from open_world.simulation.controller import Controller

MAX_EFFORT = 100.0  # 50 | 75 | 100
INFINITE_EFFORT = -1
MIN_HOLDING = 0.01
ARMS = ["right", "left"]

OPEN_POSITION = 0.548
MAX_WIDTH = 0.090  # 02178980317034

CLOSED_POSITION = 0.00165  # 3691843276424
MAX_CLOSED_POSITION = 2 * CLOSED_POSITION

# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py
# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/utils/undo_apt.py

##################################################

BASE_FRAME = "base_footprint"  # base_link | base_footprint | odom_combined

PR2_JOINT_SAFETY_LIMITS = {
    # NOTE(caelan): directly updated in the URDF
    # Taken from /ltamp-pr2/models/pr2_description/pr2.urdf
    # safety_controller: (soft_lower_limit, soft_upper_limit)
    "torso_lift_joint": (0.0115, 0.325),
    "head_pan_joint": (-2.857, 2.857),
    "head_tilt_joint": (-0.3712, 1.29626),
    "laser_tilt_mount_joint": (-0.7354, 1.43353),
    "r_shoulder_pan_joint": (-2.1353981634, 0.564601836603),
    "r_shoulder_lift_joint": (-0.3536, 1.2963),
    "r_upper_arm_roll_joint": (-3.75, 0.65),
    "r_elbow_flex_joint": (-2.1213, -0.15),
    "r_wrist_flex_joint": (-2.0, -0.1),
    "r_gripper_joint": (-0.01, 0.088),
    "l_shoulder_pan_joint": (-0.564601836603, 2.1353981634),
    "l_shoulder_lift_joint": (-0.3536, 1.2963),
    "l_upper_arm_roll_joint": (-0.65, 3.75),
    "l_elbow_flex_joint": (-2.1213, -0.15),
    "l_wrist_flex_joint": (-2.0, -0.1),
    "l_gripper_joint": (-0.01, 0.088),
    # TODO: custom base rotation limit to prevent power cord strangulation
    # TODO: could also just change joints in the URDF
    # 'theta': (-2*np.pi, 2*np.pi),
}

JOINTS = [
    "fl_caster_rotation_joint",
    "fl_caster_l_wheel_joint",
    "fl_caster_r_wheel_joint",
    "fr_caster_rotation_joint",
    "fr_caster_l_wheel_joint",
    "fr_caster_r_wheel_joint",
    "bl_caster_rotation_joint",
    "bl_caster_l_wheel_joint",
    "bl_caster_r_wheel_joint",
    "br_caster_rotation_joint",
    "br_caster_l_wheel_joint",
    "br_caster_r_wheel_joint",
    "torso_lift_joint",
    "torso_lift_motor_screw_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "laser_tilt_mount_joint",
    "r_upper_arm_roll_joint",
    "r_shoulder_pan_joint",
    "r_shoulder_lift_joint",
    "r_forearm_roll_joint",
    "r_elbow_flex_joint",
    "r_wrist_flex_joint",
    "r_wrist_roll_joint",
    "r_gripper_joint",
    "r_gripper_l_finger_joint",
    "r_gripper_r_finger_joint",
    "r_gripper_r_finger_tip_joint",
    "r_gripper_l_finger_tip_joint",
    "r_gripper_motor_screw_joint",
    "r_gripper_motor_slider_joint",
    "l_upper_arm_roll_joint",
    "l_shoulder_pan_joint",
    "l_shoulder_lift_joint",
    "l_forearm_roll_joint",
    "l_elbow_flex_joint",
    "l_wrist_flex_joint",
    "l_wrist_roll_joint",
    "l_gripper_joint",
    "l_gripper_l_finger_joint",
    "l_gripper_r_finger_joint",
    "l_gripper_r_finger_tip_joint",
    "l_gripper_l_finger_tip_joint",
    "l_gripper_motor_screw_joint",
    "l_gripper_motor_slider_joint",
]

##################################################

TORSO_JOINT_NAME = "torso_lift_joint"
HEAD_JOINT_NAMES = ["head_pan_joint", "head_tilt_joint"]

ARM_TEMPLATES = [
    "{}_shoulder_pan_joint",
    "{}_shoulder_lift_joint",
    "{}_upper_arm_roll_joint",
    "{}_elbow_flex_joint",
    "{}_forearm_roll_joint",
    "{}_wrist_flex_joint",
    "{}_wrist_roll_joint",
]

GRIPPER_TEMPLATE = "{}_gripper_joint"


def get_arm_prefix(arm):
    prefix = arm[0].lower()
    assert prefix in ["l", "r"], "Arm not found"
    return prefix


def get_gripper_joint_name(arm):
    return GRIPPER_TEMPLATE.format(get_arm_prefix(arm))


def get_arm_joint_names(arm):
    return [arm_template.format(get_arm_prefix(arm)) for arm_template in ARM_TEMPLATES]


##################################################

Client = namedtuple("Client", ["topic", "action"])


def get_client_status(client):
    status_msg = client.action_client.last_status_msg
    if status_msg is None:
        return status_msg  # Does this ever happen
    return status_msg.status_list[-1].status


def is_client_active(client):
    # http://docs.ros.org/en/indigo/api/actionlib/html/classactionlib_1_1action__client_1_1ActionClient.html#a1438c5bd0790f424ef96baf4df2574ec
    status = get_client_status(client)
    return (
        (not rospy.is_shutdown())
        and (status is not None)
        and (status < GoalStatus.SUCCEEDED)
    )


##################################################


class PR2Controller(Controller):
    # simple_clients = {
    #     'torso': Client('torso_controller/position_joint_action', SingleJointPositionAction),
    #     'head': Client('head_traj_controller/joint_trajectory_action', JointTrajectoryAction),
    # }
    # for arm in ARMS:
    #     arm = get_arm_prefix(arm)
    #     simple_clients.update({
    #         '{}_joint'.format(arm): Client('{}_arm_controller/joint_trajectory_action'.format(arm), JointTrajectoryAction),
    #         '{}_gripper'.format(arm): Client('{}_gripper_controller/gripper_action'.format(arm), Pr2GripperCommandAction),
    #         '{}_gripper_event'.format(arm): Client('{}_gripper_sensor_controller/event_detector'.format(arm), PR2GripperEventDetectorAction),
    #         '{}_grab'.format(arm): Client('{}_gripper_controller/grab'.format(arm), PR2GripperGrabAction),
    #         '{}_release'.format(arm): Client('{}_gripper_controller/grab'.format(arm), PR2GripperReleaseAction),
    #     })

    INIT_LOGS = True
    ERROR_LOGS = True
    COMMAND_LOGS = True
    """
    ============================================================================
                    Initializing all controllers    
    ============================================================================
    """

    def __init__(self, robot, client, arms=ARMS, verbose=True, **kwargs):
        super(PR2Controller, self).__init__(**kwargs)
        self.arms = tuple(arms)
        self.rate = rospy.Rate(10)

        self.clients = {}
        self.set_points = {}

        try:
            import tf  # ImportError: dynamic module does not define init function (PyInit__tf2)

            self.tf_listener = tf.TransformListener()
        except ImportError:
            self.tf_listener = None

        # Not convinced that this is actually working
        for arm in self.arms:
            client_name = "{}_joint".format(get_arm_prefix(arm))
            client = self.simple_clients[client_name]
            # goal_param = '{}_node/constraints/goal_time'.format(client.topic)
            # rospy.set_param(goal_param, 1.0)
            for joint_name in get_arm_joint_names(arm):
                joint_param = "{}_node/constraints/{}/goal".format(
                    client.topic, joint_name
                )
                rospy.set_param(joint_param, 1e-3)

        if self.INIT_LOGS and verbose:
            rospy.loginfo("starting simple action clients")
        for client_name in self.simple_clients:
            self.clients[client_name] = SimpleActionClient(
                *self.simple_clients[client_name]
            )
            if self.INIT_LOGS and verbose:
                rospy.loginfo("{} client started".format(client_name))

        for client_name in self.clients:
            result = self.clients[client_name].wait_for_server(rospy.Duration(0.1))
            self.clients[client_name].cancel_all_goals()
            if result:
                if self.INIT_LOGS and verbose:
                    rospy.loginfo("{} done initializing".format(client_name))
            else:
                if self.ERROR_LOGS:
                    rospy.loginfo("Failed to start {}".format(client_name))

        if self.INIT_LOGS:
            rospy.loginfo("Subscribing to state messages")

        self.joint_state = None
        self.joint_sub = rospy.Subscriber(
            "joint_states", JointState, self.joint_callback
        )

        # Base control copied from teleop_base_head.py
        # Consider switching this to a simple action client to match the others
        self.base_pub = rospy.Publisher("base_controller/command", Twist, queue_size=1)
        self.base_speed = rospy.get_param("~speed", 0.5)  # Run: 1.0
        self.base_turn = rospy.get_param("~turn", 1.0)  # Run: 1.5

        self.display_trajectory_pub = rospy.Publisher(
            "ros_controller/display_trajectory", DisplayTrajectory, queue_size=1
        )
        self.robot_state_pub = rospy.Publisher(
            "ros_controller/robot_state", DisplayRobotState, queue_size=1
        )
        # self.sound_pub = rospy.Publisher('robotsound', SoundRequest, queue_size=1)

        self.gripper_events = {}
        self.gripper_event_subs = {
            arm: rospy.Subscriber(
                "/{}_gripper_sensor_controller/event_detector_state".format(
                    get_arm_prefix(arm)
                ),
                PR2GripperEventDetectorData,
                self.get_gripper_event_callback(arm),
            )
            for arm in self.arms
        }

        self.wait_until_ready()
        if self.INIT_LOGS and verbose:
            rospy.loginfo("Done initializing PR2 Controller!")

    ##################################################

    def speak(self, phrase):
        # TODO: upgrade to python3
        #   File "/opt/ros/indigo/lib/sound_play/say.py", line 44
        #     print 'Usage: %s \'String to say.\''%sys.argv[0]
        os.system('rosrun sound_play say.py "{}"'.format(phrase))
        # request = SoundRequest()
        # self.sound_pub.publish(request)

    def rest_for_duration(self, duration):
        time.sleep(duration)

    """
    =============================================================== #XXX make these all nice :)
                    State subscriber callbacks    
    ===============================================================
    """

    def get_robot_state(self):
        # pose = pose_from_trans(self.get_world_pose(BASE_FRAME)) # TODO: could get this transform directly
        # transform = Transform(Vector3(*point_from_pose(pose)), Quaternion(*quat_from_pose(pose)))
        # transform = self.get_transform()
        # if transform is None:
        #    return None
        state = RobotState()
        state.joint_state = self.joint_state
        # state.multi_dof_joint_state.header.frame_id = '/base_footprint'
        # state.multi_dof_joint_state.header.stamp = rospy.Time(0)
        # state.multi_dof_joint_state.joints = ['world_joint']
        # state.multi_dof_joint_state.transforms = [transform]
        # 'world_joint'
        # http://cram-system.org/tutorials/intermediate/moveit
        state.attached_collision_objects = []
        state.is_diff = False
        # rostopic info /joint_states
        return state

    def get_display_trajectory(self, *joint_trajectories):
        display_trajectory = DisplayTrajectory()
        display_trajectory.model_id = "pr2"
        for joint_trajectory in joint_trajectories:
            robot_trajectory = RobotTrajectory()
            robot_trajectory.joint_trajectory = joint_trajectory
            # robot_trajectory.multi_dof_joint_trajectory = ...
            display_trajectory.trajectory.append(robot_trajectory)
        display_trajectory.trajectory_start = self.get_robot_state()
        return display_trajectory

    def publish_joint_trajectories(self, *joint_trajectories):
        display_trajectory = self.get_display_trajectory(*joint_trajectories)
        display_state = DisplayRobotState()
        display_state.state = display_trajectory.trajectory_start
        # self.robot_state_pub.publish(display_state)
        self.display_trajectory_pub.publish(display_trajectory)
        # raw_input('Continue?')

        last_trajectory = joint_trajectories[-1]
        last_conf = last_trajectory.points[-1].positions
        joint_state = display_state.state.joint_state
        joint_state.position = list(joint_state.position)
        for joint_name, position in zip(last_trajectory.joint_names, last_conf):
            joint_index = joint_state.name.index(joint_name)
            joint_state.position[joint_index] = position
        self.robot_state_pub.publish(display_state)
        # TODO: record executed trajectory and overlay them
        return display_trajectory

    ##################################################

    def not_ready(self):
        return self.joint_state is None

    def wait_until_ready(self, timeout=5.0):
        end_time = rospy.Time.now() + rospy.Duration(timeout)
        while (
            not rospy.is_shutdown()
            and (rospy.Time.now() < end_time)
            and self.not_ready()
        ):
            self.rate.sleep()
        if self.not_ready():
            if self.ERROR_LOGS:
                rospy.loginfo("Warning! Did not complete subscribing")
        else:
            if self.INIT_LOGS:
                rospy.loginfo("Robot ready")

    def joint_callback(self, data):
        self.joint_state = data

    def wait(self, duration):
        rospy.sleep(duration)

    """
    ===============================================================
                   Get State information    
    ===============================================================
    """

    @property
    def joint_names(self):
        # if self.not_ready():
        #     return None
        return list(self.joint_state.name)  # sorted

    @property
    def joint_positions(self):
        return dict(zip(self.joint_state.name, self.joint_state.position))

    @property
    def joint_velocities(self):
        return dict(zip(self.joint_state.name, self.joint_state.velocity))

    def is_closed_position(self, position):
        return position <= MAX_CLOSED_POSITION

    def is_arm_fully_closed(self, arm):
        gripper_joint = get_gripper_joint_name(arm)
        return self.is_closed_position(self.joint_positions[gripper_joint])
        # if self.not_ready() or (gripper_joint not in self.set_points):
        #     return True
        # return self.is_closed_position(self.set_points[gripper_joint]) and \
        #        self.is_closed_position(self.joint_positions[gripper_joint])

    def any_arm_fully_closed(self):
        return any(map(self.is_arm_fully_closed, self.arms))

    # return the current Cartesian pose of the gripper
    def return_cartesian_pose(self, arm, frame=BASE_FRAME):
        assert self.tf_listener is not None
        import tf

        end_time = rospy.Time.now() + rospy.Duration(5)
        link = "{}_gripper_tool_frame".format(arm)
        while not rospy.is_shutdown() and (rospy.Time.now() < end_time):
            try:
                t = self.tf_listener.getLatestCommonTime(frame, link)
                (trans, rot) = self.tf_listener.lookupTransform(frame, link, t)
                # if frame == 'base_link' and self.COMMAND_LOGS:
                #    expected_trans, expected_rot = arm_fk(arm, self.get_arm_positions(arm), self.get_torso_position())
                #    error_threshold = 0.05 # 5 cm position difference
                #    if any([abs(t - e) > error_threshold for t, e in zip(trans, expected_trans)]):
                #        rospy.loginfo("TF position does not match FK position")
                #        rospy.loginfo("TF Pose: " + str([trans, rot]))
                #        rospy.loginfo("FK Pose: " + str([expected_trans, expected_rot]))
                return list(trans), list(rot)
            except (tf.Exception, tf.ExtrapolationException):
                rospy.sleep(0.5)
                # current_time = rospy.get_rostime()
                if self.COMMAND_LOGS:
                    rospy.logerr(
                        "Waiting for a tf transform between {} and {}".format(
                            frame, link
                        )
                    )
        if self.ERROR_LOGS:
            rospy.logerr(
                "Return_cartesian_pose waited 10 seconds tf transform! Returning None"
            )
        return None, None

    """
    ===============================================================
                Send Commands for Action Clients                
    ===============================================================
    """

    def wait_for_clients(self, clients, timeout=None):
        end_time = rospy.Time.now() + rospy.Duration(timeout)
        while (
            (not rospy.is_shutdown())
            and (rospy.Time.now() < end_time)
            and any(is_client_active(client) for client in clients)
        ):
            self.rate.sleep()
        return not rospy.is_shutdown() and (rospy.Time.now() < end_time)

    def _send_command(
        self, client_name, goal, blocking=False, timeout=None, buffer=0.1
    ):
        if client_name not in self.clients:
            return False
        client = self.clients[client_name]
        client.send_goal(goal)
        start_time = rospy.Time.now()
        # rospy.loginfo(goal)
        rospy.sleep(buffer)
        if self.COMMAND_LOGS:
            rospy.loginfo("Command sent to {} client".format(client_name))
        if not blocking:  # XXX why isn't this perfect?
            # return None
            return client

        self.wait_for_clients(
            clients=[client], timeout=timeout + buffer
        )  # send_goal_and_wait

        # It's reporting time outs that are too early
        # FIXED: See comments about rospy.Time(0)
        status = get_client_status(client)
        text = GoalStatus.to_string(status)
        if GoalStatus.SUCCEEDED <= status:
            if self.COMMAND_LOGS:
                rospy.loginfo("Goal status {} achieved. Exiting".format(text))
        else:
            if self.ERROR_LOGS:
                rospy.loginfo("Ending due to timeout {}".format(text))

        result = client.get_result()
        elapsed_time = (rospy.Time.now() - start_time).to_sec()
        print(
            "Executed in {:.3f} seconds. Predicted to take {:.3f} seconds.".format(
                elapsed_time, timeout
            )
        )
        # state = client.get_state()
        # print('Goal state {}'.format(GoalStatus.to_string(state)))
        # get_goal_status_text
        return result

    def command_base(self, x, y, yaw):
        # This doesn't use _send_command because the base uses a different publisher. Probably could be switched later.
        # Don't forget that x, y, and yaw are multiplied by self.base_speed and self.base_turn.
        # Recommended values for x, y, and yaw are {-1, 0, 1}. Consider cutting out the speed multipliers.
        # The motion is pretty jerky. Consider sending a smoother acceleration trajectory.
        # We can also just re-register using perception after it stops.
        # +x is forward, -x is backward, +y is left, -y is right, +yaw is ccw looking down from above, -yaw is cw
        # Once the robot gets sent a base command, it continues on that velocity. Remember to send a stop command.
        twist = Twist(
            linear=Vector3(x=x * self.base_speed, y=y * self.base_speed, z=0),
            angular=Vector3(x=0, y=0, z=yaw * self.base_turn),
        )
        self.base_pub.publish(twist)

    def stop_base(self):
        self.command_base(x=0, y=0, yaw=0)

    def command_torso(self, position, timeout, blocking=True):
        # TODO: raise/lower torso
        goal = SingleJointPositionGoal(
            position=position, min_duration=rospy.Duration(timeout), max_velocity=1.0
        )
        self.set_points.update({TORSO_JOINT_NAME: position})

        client_name = "torso"
        return self._send_command(client_name, goal, blocking=blocking, timeout=timeout)

    def command_head(self, angles, timeout, blocking=True):
        goal = JointTrajectoryGoal()
        goal.trajectory.joint_names = HEAD_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = angles
        point.time_from_start = rospy.Duration(timeout)
        goal.trajectory.points.append(point)
        self.set_points.update(
            zip(goal.trajectory.joint_names, goal.trajectory.points[-1].positions)
        )

        client = self.clients["head"]
        if blocking:
            client.send_goal_and_wait(goal)
        else:
            client.send_goal(goal)
        return client

    def _send_command2(self, client_name, goal, timeout=2.0, blocking=True):
        client = self.clients[client_name]
        client.send_goal(goal)
        if blocking:
            client.wait_for_result(rospy.Duration(timeout))
        return client

    def command_grab(self, arm, hardness_gain=0.03, **kwargs):
        # http://docs.ros.org/en/fuerte/api/assistive_teleop/html/pr2__reactive__grippers_8py_source.html
        arm = get_arm_prefix(arm)
        goal = PR2GripperGrabGoal()
        goal.command.hardness_gain = hardness_gain
        self.set_points.update({get_gripper_joint_name(arm): 0.0})

        client_name = "{}_grab".format(arm)
        # return self._send_command(client_name, goal, blocking=blocking, timeout=timeout)
        return self._send_command2(client_name, goal, **kwargs)

    def command_release(self, arm, **kwargs):
        # http://docs.ros.org/en/fuerte/api/assistive_teleop/html/pr2__reactive__grippers_8py_source.html
        arm = get_arm_prefix(arm)
        goal = PR2GripperReleaseGoal()
        goal.command.release_event.trigger_conditions = 0
        goal.command.release_event.acceleration_trigger_magnitude = 3
        goal.command.release_event.slip_trigger_magnitude = 0.005

        client_name = "{}_release".format(arm)
        # return self._send_command(client_name, goal, blocking=blocking, timeout=timeout)
        return self._send_command2(client_name, goal, **kwargs)

    # Sending a negative max_effort means no limit for maximum effort.
    def command_gripper(
        self, arm, position, max_effort=MAX_EFFORT, timeout=2.0, blocking=True
    ):
        # return self.command_grab(arm, timeout=timeout, blocking=blocking)
        arm = get_arm_prefix(arm)
        goal = Pr2GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        self.set_points.update({get_gripper_joint_name(arm): position})

        client_name = "{}_gripper".format(arm)
        return self._send_command(client_name, goal, blocking=blocking, timeout=timeout)

    def open_gripper(self, arm, **kwargs):
        return self.command_gripper(arm, position=OPEN_POSITION, **kwargs)

    def close_gripper(self, arm, **kwargs):
        return self.command_gripper(arm, position=0.0, **kwargs)

    """
    ===============================================================
                    Joint Control Commands 
    ===============================================================
    """

    def command_arm_trajectory(
        self, arm, angles, times_from_start, blocking=True, time_buffer=5.0
    ):
        # angles is a list of joint angles, times is a list of times from start
        # When calling joints on an arm, needs to be called with all the angles in the arm
        # rospy.Duration is fine with taking floats, so the times can be floats
        assert len(angles) == len(times_from_start)

        arm = get_arm_prefix(arm)
        goal = JointTrajectoryGoal()
        goal.trajectory.joint_names = get_arm_joint_names(arm)
        for positions, time_from_start in zip(angles, times_from_start):
            point = JointTrajectoryPoint()
            point.positions = positions
            # point.velocities = [(ang[i] - last_position[i])/(t - last_time) for i in range(7)]
            # point.velocities = [0.0 for _ in range(len(positions))]
            point.time_from_start = rospy.Duration(time_from_start)
            goal.trajectory.points.append(point)

        # goal.trajectory.header.stamp = rospy.Time.now()
        # Using rospy.Time.now() is bad because the PR2 might be a few seconds ahead.
        # In that case, it clips off the first few points in the trajectory.
        # The clipping causes a jerking motion which can ruin the motion.
        goal.trajectory.header.stamp = rospy.Time(0)
        self.publish_joint_trajectories(goal.trajectory)
        # TODO(caelan): multiplicative time_buffer (dialation)
        timeout = times_from_start[-1] + time_buffer
        self.set_points.update(
            zip(goal.trajectory.joint_names, goal.trajectory.points[-1].positions)
        )

        result = self._send_command(
            "{}_joint".format(arm), goal, blocking=blocking, timeout=timeout
        )

        actual = np.array(
            [
                self.joint_positions[joint_name]
                for joint_name in goal.trajectory.joint_names
            ]
        )
        desired = np.array(goal.trajectory.points[-1].positions)
        print(
            "Error:",
            list(zip(goal.trajectory.joint_names, np.round(actual - desired, 5))),
        )
        return result

    def command_arm(self, arm, angles, timeout, **kwargs):
        return self.command_arm_trajectory(arm, [angles], [timeout], **kwargs)

    def stop_arm(self, arm):
        goal = JointTrajectoryGoal()
        goal.trajectory.joint_names = get_arm_joint_names(arm)
        # goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.header.stamp = rospy.Time(0)
        return self._send_command("{}_joint".format(arm), goal, blocking=False)

    def command_group(
        self, group, positions, timeout, **kwargs
    ):  # TODO: default timeout
        if group == "head":
            return self.command_head(positions, timeout, **kwargs)
        elif group == "torso":
            [position] = positions
            return self.command_torso(position, timeout, **kwargs)
        elif group.endswith("_arm"):
            return self.command_arm(get_arm_prefix(group), positions, timeout, **kwargs)
        elif group.endswith("_gripper"):
            position = np.average(positions)
            return self.command_gripper(
                get_arm_prefix(group), position, **kwargs
            )  # timeout,
        raise NotImplementedError(group)

    def command_group_trajectory(
        self, group, positions, times_from_start, blocking=True, **kwargs
    ):
        if group.endswith("_arm"):  # TODO: head
            return self.command_arm_trajectory(
                get_arm_prefix(group),
                positions,
                times_from_start,
                blocking=blocking,
                **kwargs
            )
        return self.command_group(
            group, positions[-1], times_from_start[-1], blocking=blocking, **kwargs
        )

    ##################################################

    def get_gripper_event_callback(self, arm):
        arm = get_arm_prefix(arm)

        def gripper_event_callback(data):
            self.gripper_events[arm] = data

        return gripper_event_callback

    def get_gripper_event(self, arm):
        # This may not work until you subscribe to the gripper event
        arm = get_arm_prefix(arm)
        if arm in self.gripper_events:
            msg = self.gripper_events[arm]
            event = msg.trigger_conditions_met or msg.acceleration_event
            return event
        print("No gripper event found... did you launch gripper sensor action?")
        return None

    def command_event_detector(self, arm, trigger, magnitude=4.0, **kwargs):
        # http://wiki.ros.org/pr2_gripper_sensor_action
        # http://wiki.ros.org/pr2_gripper_sensor_msgs
        # http://wiki.ros.org/pr2_gripper_sensor_controller?distro=melodic
        # http://docs.ros.org/en/melodic/api/pr2_gripper_sensor_msgs/html/index-msg.html
        # https://docs.ros.org/en/diamondback/api/pr2_gripper_sensor_msgs/html/classpr2__gripper__sensor__msgs_1_1msg_1_1__PR2GripperForceServoFeedback_1_1PR2GripperForceServoFeedback.html
        # http://wiki.ros.org/pr2_gripper_sensor_action/Tutorials/Grab%20and%20Release%20an%20Object%20Using%20pr2_gripper_sensor_action
        # https://github.mit.edu/caelan/mudfish/blob/4a3120ad591510fa7c16139a1411cb0ab72f39a8/scripts/controller.py#L493
        # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/voxel_detection.py#L228
        goal = PR2GripperEventDetectorGoal()

        trigger = goal.command.ACC
        # trigger = goal.command.SLIP_AND_ACC
        # trigger = goal.command.FINGER_SIDE_IMPACT_OR_SLIP_OR_ACC
        # trigger = goal.command.SLIP

        goal.command.trigger_conditions = trigger
        goal.command.acceleration_trigger_magnitude = magnitude
        return self._send_command("{}_gripper_event".format(arm), goal, **kwargs)
