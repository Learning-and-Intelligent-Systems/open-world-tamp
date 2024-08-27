import math
import time

import numpy as np
from pybullet_tools.retime import sample_curve
from pybullet_tools.utils import (GRAVITY, INF, add_pose_constraint,
                                  control_joints, draw_pose, elapsed_time,
                                  enable_gravity, get_distance,
                                  get_duration_fn, get_joint_positions,
                                  get_joint_velocities, get_movable_joints,
                                  get_pose, get_time_step, has_gui,
                                  inf_generator, interpolate,
                                  interpolate_poses, joint_controller,
                                  point_from_pose, quat_angle_between,
                                  quat_from_pose, remove_constraint,
                                  remove_handles, set_joint_positions,
                                  step_simulation, wait_if_gui,
                                  waypoint_joint_controller)


def step_curve(body, joints, curve, time_step=2e-2, print_freq=None, **kwargs):
    start_time = time.time()
    num_steps = 0
    time_elapsed = 0.0
    last_print = time_elapsed
    for num_steps, (time_elapsed, positions) in enumerate(
        sample_curve(curve, time_step=time_step)
    ):
        set_joint_positions(body, joints, positions, **kwargs)
        # num_steps += 1
        # time_elapsed += time_step
        if (print_freq is not None) and (print_freq <= (time_elapsed - last_print)):
            print(
                "Step: {} | Sim secs: {:.3f} | Real secs: {:.3f} | Steps/sec {:.3f}".format(
                    num_steps,
                    time_elapsed,
                    elapsed_time(start_time),
                    num_steps / elapsed_time(start_time),
                )
            )
            last_print = time_elapsed
        yield positions
    if print_freq is not None:
        print(
            "Simulated {} steps ({:.3f} sim seconds) in {:.3f} real seconds".format(
                num_steps, time_elapsed, elapsed_time(start_time)
            )
        )
    # return time_elapsed


def project_curve(current_positions, curve, time_step=1e-3):
    # TODO: see cognitive-architectures
    raise NotImplementedError("Project from the positions to the nearest point")
    _, d = np.shape(curve.y)
    assert len(current_positions) == d
    timed_samples = list(sample_curve(curve, time_step=time_step))
    # duration_fn = get_duration_fn(body, joints, velocities=max_velocities) # TODO: duration
    control_time, positions = min(
        timed_samples, key=lambda item: get_distance(current_positions, item[1])
    )
    # interpolate_path(self.robot, self.joints, path)
    print(control_time, positions)


#######################################################


def simulate_controller(controller, real_per_sim=2.0, print_freq=0.1, hook=None):
    # TODO: multiple controllers
    start_time = time.time()
    dt = get_time_step()
    num_steps = 0
    time_elapsed = 0.0
    last_print = time_elapsed
    enable_gravity()
    for _ in controller:
        step_simulation()
        if not (hook is None) and num_steps % 10 == 0:
            hook()
        num_steps += 1
        time_elapsed += dt
        if (print_freq is not None) and (print_freq <= (time_elapsed - last_print)):
            print(
                "Step: {} | Sim secs: {:.3f} | Real secs: {:.3f} | Steps/sec {:.3f}".format(
                    num_steps,
                    time_elapsed,
                    elapsed_time(start_time),
                    num_steps / elapsed_time(start_time),
                )
            )
            last_print = time_elapsed
        if has_gui():
            if real_per_sim is None:
                wait_if_gui()
            else:
                # TODO: adjust based on the frame rate
                # duration = real_per_sim # real per step
                duration = real_per_sim * dt  # real per second
                time.sleep(duration)
                # wait_for_duration(duration)
    if print_freq is not None:
        print(
            "Simulated {} steps ({:.3f} sim seconds) in {:.3f} real seconds".format(
                num_steps, time_elapsed, elapsed_time(start_time)
            )
        )
    return time_elapsed


#######################################################


def constant_controller(value=None):
    return (value for _ in inf_generator())


def timeout_controller(controller, timeout=INF, time_step=None):
    if time_step is None:
        time_step = get_time_step()
    time_elapsed = 0.0
    for output in controller:
        if time_elapsed > timeout:
            break
        yield output
        time_elapsed += time_step


def stall_for_duration(duration):
    dt = get_time_step()
    time_elapsed = 0.0
    while time_elapsed < duration:
        yield time_elapsed
        # step_simulation()
        time_elapsed += dt
    # return time_elapsed


def hold_for_duration(body, duration, joints=None, **kwargs):
    if joints is None:
        joints = get_movable_joints(body)
    control_joints(body, joints, **kwargs)
    return stall_for_duration(duration)


def pose_controller(
    body,
    target_pose,
    pos_tol=1e-3,
    ori_tol=math.radians(1),
    max_force=100 * GRAVITY,
    timeout=INF,
    verbose=False,
):
    # TODO: with statement hold pose
    constraint = add_pose_constraint(body, pose=target_pose, max_force=max_force)
    target_pos, target_quat = target_pose
    dt = get_time_step()
    time_elapsed = 0.0
    while time_elapsed < timeout:
        pose = get_pose(body)
        pos_error = get_distance(point_from_pose(pose), target_pos)
        ori_error = quat_angle_between(quat_from_pose(pose), target_quat)
        if verbose:
            print(
                "Position error: {:.3f} | Orientation error: {:.3f}".format(
                    pos_error, ori_error
                )
            )
        if (pos_error <= pos_tol) and (ori_error <= ori_tol):
            break
        yield pose
        time_elapsed += dt
    remove_constraint(constraint)


def interpolate_pose_controller(body, target_pose, timeout=INF, draw=True, **kwargs):
    # TODO: interpolate using velocities instead
    pose_waypoints = list(
        interpolate_poses(
            get_pose(body),
            target_pose,
            pos_step_size=0.01,
            ori_step_size=math.radians(1),
        )
    )
    dt = get_time_step()
    time_elapsed = 0.0
    handles = []
    for num, waypoint_pose in enumerate(pose_waypoints):
        if time_elapsed >= timeout:
            break
        # print('Waypoint {}'.format(num))
        is_goal = num == len(pose_waypoints) - 1
        if draw:
            handles.extend(draw_pose(waypoint_pose, length=0.05))
        for output in pose_controller(body, waypoint_pose, **kwargs):
            yield output
            time_elapsed += dt
        # wait_if_gui()
        remove_handles(handles)


def interpolate_controller(
    body, joints, target_positions, max_velocities=None, dt=1e-1, timeout=INF, **kwargs
):
    duration_fn = get_duration_fn(body, joints, velocities=max_velocities)
    positions = get_joint_positions(body, joints)
    duration = duration_fn(positions, target_positions)
    num_steps = int(np.ceil(duration / dt))
    waypoints = list(interpolate(positions, target_positions, num_steps))
    time_elapsed = 0.0
    for num, waypoint in enumerate(waypoints):
        if time_elapsed >= timeout:
            break
        is_goal = num == len(waypoints) - 1
        # handles = []
        for output in joint_controller(
            body, joints, waypoint, timeout=(timeout - time_elapsed), **kwargs
        ):
            yield output
            time_elapsed += get_time_step()
        # remove_handles(handles)


#######################################################


def follow_path(
    body,
    joints,
    path,
    waypoint_tol=1e-2 * np.pi,
    goal_tol=5e-3 * np.pi,
    waypoint_timeout=1.0,
    path_timeout=INF,
    lead_step=0.1,
    verbose=True,
    **kwargs
):
    start_time = time.time()
    dt = get_time_step()
    handles = []
    steps = 0
    duration = 0.0
    odometry = [
        np.array(get_joint_positions(body, joints))
    ]  # TODO: plot the comparison with the nominal trajectory
    control_joints(
        body, get_movable_joints(body)
    )  # TODO: different (infinite) gains for hold joints
    for num, positions in enumerate(path):
        if duration > path_timeout:
            break
        # start = duration
        is_goal = num == len(path) - 1
        tolerance = goal_tol if is_goal else waypoint_tol

        # TODO: velocity control is ineffective
        # velocities = np.zeros(len(joints))
        # velocities = velocities_curve(control_time)
        # for joint, position, velocity in zip(joints, positions, velocities):
        #     control_joint(body, joint, position, velocity)
        # control_joints(body, joints, positions, velocities,) # position_gain=1e-1)
        # handles.extend(draw_pose(pose_from_pose2d(positions, z=1e-2), length=5e-2))

        if verbose:
            print(
                "Waypoint: {} | Goal: {} | Sim steps: {} | Sim secs: {:.3f} | Steps/sec {:.3f}".format(
                    num, is_goal, steps, duration, steps / elapsed_time(start_time)
                )
            )

        # TODO: adjust waypoint_timeout based on the duration
        if lead_step is None:
            controller = joint_controller(
                body,
                joints,
                positions,
                tolerance=tolerance,
                timeout=waypoint_timeout,
                **kwargs
            )
        else:
            controller = waypoint_joint_controller(
                body,
                joints,
                positions,
                tolerance=tolerance,
                time_step=lead_step,
                timeout=waypoint_timeout,
                **kwargs
            )
        for output in controller:
            yield output
            # step_simulation()
            # wait_if_gui()
            # wait_for_duration(10*dt)
            steps += 1
            duration += dt
            odometry.append(
                odometry[-1] + dt * np.array(get_joint_velocities(body, joints))
            )
        remove_handles(handles)
    if verbose:
        print(
            "Followed {} waypoints in {} sim steps ({:.3f} sim seconds)".format(
                len(path), steps, duration
            )
        )


def follow_curve(body, joints, positions_curve, time_step=1e-1, **kwargs):
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pdControl.py
    control_times = np.append(
        np.arange(positions_curve.x[0], positions_curve.x[-1], step=time_step),
        [positions_curve.x[-1]],
    )
    # TODO: sample_curve
    # velocities_curve = positions_curve.derivative()
    path = [positions_curve(control_time) for control_time in control_times]
    return follow_path(body, joints, path, **kwargs)
