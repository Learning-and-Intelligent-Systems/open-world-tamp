import random

import numpy as np
from open_world.planning.primitives import GroupConf, GroupTrajectory, Sequence
from open_world.planning.pushing import cartesian_path_collision
from open_world.planning.samplers import plan_workspace_motion
from pybullet_tools.pr2_utils import side_from_arm

import owt.pb_utils as pbu

LIQUID_QUAT = quat_from_euler(Euler(pitch=np.pi))
RELATIVE_POUR = True
RELATIVE_POUR_SCALING = {
    "axis_in_cup_x": "cup_diameter",
    "axis_in_cup_z": "cup_height",
    "axis_in_bowl_x": "bowl_diameter",
    #'axis_in_bowl_z': 'bowl_height',
}


def compute_base_diameter(vertices, epsilon=0.001):
    lower = np.min(vertices, axis=0)
    threshold = lower[2] + epsilon
    base_vertices = [vertex for vertex in vertices if vertex[2] <= threshold]
    base_aabb = aabb_from_points(base_vertices)
    return np.average(get_aabb_extent(base_aabb)[:2])


def get_pour_feature(robot, environment, bowl_body, cup_body):
    bowl_reference = unit_pose()
    _, (bowl_d, bowl_h) = approximate_as_cylinder(bowl_body, body_pose=bowl_reference)
    bowl_vertices = vertices_from_rigid(bowl_body)

    cup_reference = (unit_point(), LIQUID_QUAT)
    _, (cup_d, _, cup_h) = approximate_as_prism(cup_body, body_pose=cup_reference)
    cup_vertices = vertices_from_rigid(cup_body)

    # TODO: compute moments/other features from the mesh
    feature = {
        "bowl_diameter": bowl_d,
        "bowl_height": bowl_h,
        "bowl_base_diameter": compute_base_diameter(bowl_vertices),
        "cup_diameter": cup_d,
        "cup_height": cup_h,
        "cup_base_diameter": compute_base_diameter(cup_vertices),
    }
    return feature


def sample_pour_parameter(robot, environemnt, feature):
    # TODO: adjust for RELATIVE_POUR
    cup_pour_pitch = -3 * np.pi / 4
    # pour_cup_pitch = -5*np.pi/6
    # pour_cup_pitch = -np.pi

    # axis_in_cup_center_x = -0.05
    axis_in_cup_center_x = 0
    # axis_in_cup_center_z = -feature['cup_height']/2.
    axis_in_cup_center_z = 0.0  # This is in meters (not a fraction of the high)
    # axis_in_cup_center_z = feature['cup_height']/2.

    # tl := top left | tr := top right
    cup_tl_in_center = np.array(
        [-feature["cup_diameter"] / 2, 0, feature["cup_height"] / 2]
    )
    cup_tl_in_axis = cup_tl_in_center - Point(
        x=axis_in_cup_center_x, z=axis_in_cup_center_z
    )
    cup_tl_angle = np.math.atan2(cup_tl_in_axis[2], cup_tl_in_axis[0])
    cup_tl_pour_pitch = cup_pour_pitch - cup_tl_angle

    cup_radius2d = np.linalg.norm([cup_tl_in_axis])
    pivot_in_bowl_tr = Point(
        x=-(cup_radius2d * np.math.cos(cup_tl_pour_pitch) + 0.01),
        z=(cup_radius2d * np.math.sin(cup_tl_pour_pitch) + 0.01),
    )

    bowl_tr_in_bowl_center = Point(
        x=feature["bowl_diameter"] / 2, z=feature["bowl_height"] / 2
    )
    pivot_in_bowl_center = bowl_tr_in_bowl_center + pivot_in_bowl_tr

    parameter = {
        "pitch": cup_pour_pitch,
        "axis_in_cup_x": axis_in_cup_center_x,
        "axis_in_cup_z": axis_in_cup_center_z,
        "axis_in_bowl_x": pivot_in_bowl_center[0],
        "axis_in_bowl_z": pivot_in_bowl_center[2],
        #'velocity': None,
        #'bowl_yaw': None,
        #'cup_yaw': None,
        "relative": RELATIVE_POUR,
    }
    if RELATIVE_POUR:
        parameter = scale_parameter(feature, parameter, RELATIVE_POUR_SCALING)
    yield parameter


def get_urdf_from_z_axis(body, z_fraction, reference_quat=unit_quat()):
    # AKA the pose of the body's center wrt to the body's origin
    # z_fraction=0. => bottom, z_fraction=0.5 => center, z_fraction=1. => top
    ref_from_urdf = (unit_point(), reference_quat)
    center_in_ref, (_, height) = approximate_as_cylinder(body, body_pose=ref_from_urdf)
    center_in_ref[2] += (z_fraction - 0.5) * height
    ref_from_center = (
        center_in_ref,
        unit_quat(),
    )  # Maps from center frame to origin frame
    urdf_from_center = multiply(invert(ref_from_urdf), ref_from_center)
    return urdf_from_center


def get_urdf_from_top(body, **kwargs):
    return get_urdf_from_z_axis(body, z_fraction=1.0, **kwargs)


def get_bowl_from_pivot(robot, environment, bowl, feature, parameter):
    bowl_urdf_from_center = get_urdf_from_top(
        bowl
    )  # get_urdf_from_base | get_urdf_from_center
    if RELATIVE_POUR:
        parameter = scale_parameter(
            feature, parameter, RELATIVE_POUR_SCALING, descale=True
        )
    bowl_base_from_pivot = Pose(
        Point(x=parameter["axis_in_bowl_x"], z=parameter["axis_in_bowl_z"])
    )
    return multiply(bowl_urdf_from_center, bowl_base_from_pivot)


def scale_parameter(feature, parameter, scaling={}, descale=False):
    scaled_parameter = dict(parameter)
    for param, feat in scaling.items():
        # if (feat in feature) and (param in scaled_parameter):
        if descale:
            scaled_parameter[param] *= feature[feat]
        else:
            scaled_parameter[param] /= feature[feat]
    return scaled_parameter


def constant_velocity_times(waypoints, velocity=2.0):
    """List of quaternion waypoints.

    velocity in radians/s
    @return a list of times, [0,0.1,0.2,0.3,0.4] to execute each point in the trajectory
    """
    # goes over each pair of quaternions and finds the time it would take for them to move with a constant velocity
    times_from_start = [0.0]
    waypoints_iter = iter(waypoints)
    _, prev_quat = next(waypoints_iter)
    while True:
        try:
            _, curr_quat = next(waypoints_iter)
            delta_angle = quat_angle_between(prev_quat, curr_quat)
            delta_time = delta_angle / velocity
            times_from_start.append(times_from_start[-1] + delta_time)
            prev_quat = curr_quat
        except StopIteration:
            return tuple(times_from_start)


def pour_path_from_parameter(
    robot, environment, cup_body, bowl_body, feature, parameter, cup_yaw=None
):
    # cup_urdf_from_center = get_urdf_from_center(cup_body, reference_quat=get_liquid_quat(feature['cup_name']))
    ref_from_urdf = (unit_point(), LIQUID_QUAT)
    cup_center_in_ref, _ = approximate_as_prism(cup_body, body_pose=ref_from_urdf)
    cup_center_in_ref[
        :2
    ] = 0  # Assumes the xy pour center is specified by the URDF (e.g. for spoons)
    cup_urdf_from_center = multiply(
        invert(ref_from_urdf), Pose(point=cup_center_in_ref)
    )

    # TODO: allow some deviation around cup_yaw for spoons
    if cup_yaw is None:
        cup_yaw = random.uniform(-np.pi, np.pi)
    z_rotate_cup = Pose(euler=Euler(yaw=cup_yaw))

    bowl_from_pivot = get_bowl_from_pivot(
        robot, environment, bowl_body, feature, parameter
    )
    if RELATIVE_POUR:
        parameter = scale_parameter(
            feature, parameter, RELATIVE_POUR_SCALING, descale=True
        )
    base_from_pivot = Pose(
        Point(x=parameter["axis_in_cup_x"], z=parameter["axis_in_cup_z"])
    )

    initial_pitch = 0
    final_pitch = parameter["pitch"]
    assert -np.pi <= final_pitch <= initial_pitch
    cup_path_in_bowl = []
    for pitch in list(np.arange(final_pitch, initial_pitch, np.pi / 16)) + [
        initial_pitch
    ]:
        rotate_pivot = Pose(
            euler=Euler(pitch=pitch)
        )  # Can also interpolate directly between start and end quat
        cup_path_in_bowl.append(
            multiply(
                bowl_from_pivot,
                rotate_pivot,
                invert(base_from_pivot),
                z_rotate_cup,
                invert(cup_urdf_from_center),
            )
        )
    cup_times = constant_velocity_times(cup_path_in_bowl)
    # TODO: check for collisions here?

    return cup_path_in_bowl, cup_times


def get_plan_pour_fn(
    robot,
    environment=[],
    max_samples=25,
    max_attempts=10,
    collisions=True,
    parameter_fns={},
    repeat=False,
    **kwargs
):
    environment = list(environment)
    side = robot.get_arbitrary_side()
    arm_group, gripper_group, tool_name = robot.manipulators[side]
    robot.get_component(gripper_group)

    # TODO(caelan): could also simulate the predicated sample
    # TODO(caelan): make final the orientation be aligned with gripper

    robot.disabled_collisions
    approach_tform = Pose(point=np.array([-0.1, 0, 0]))  # Tool coordinates

    def gen_fn(arm, bowl, pose, cup, grasp):
        if bowl == cup:
            return
        bowl_pose = pose.get_pose()
        # attachment = get_grasp_attachment(robot, environment, arm, grasp)
        feature = get_pour_feature(robot, environment, bowl, cup)
        for parameter in sample_pour_parameter(robot, environment, feature):
            # TODO: this may be called several times with different grasps
            for i in range(max_attempts):
                set_pose(bowl, bowl_pose)  # Reset because might have changed
                cup_path_bowl, _ = pour_path_from_parameter(
                    robot, environment, cup, bowl, feature, parameter
                )
                rotate_bowl = Pose(euler=Euler(yaw=random.uniform(-np.pi, np.pi)))
                cup_path = [
                    multiply(bowl_pose, invert(rotate_bowl), cup_pose_bowl)
                    for cup_pose_bowl in cup_path_bowl
                ]
                if cartesian_path_collision(cup, cup_path, environment + [bowl]):
                    continue

                tool_waypoints = [multiply(p, invert(grasp.grasp)) for p in cup_path]

                post_path = plan_workspace_motion(
                    robot, side, tool_waypoints, obstacles=environment, attachment=[]
                )

                if post_path is None:
                    continue

                pre_conf = GroupConf(robot, arm, positions=post_path[-1])
                pre_path = post_path[::-1]
                post_conf = pre_conf
                commands = [
                    GroupTrajectory(robot, arm_group, path=pre_path),
                    # Rest(duration=2.0),
                    GroupTrajectory(robot, arm_group, path=post_path),
                ]
                sequence = Sequence(
                    commands=commands, name="pour-{}-{}".format(side_from_arm(arm), cup)
                )

                return (pre_conf, post_conf, sequence)
        return None

    return gen_fn
