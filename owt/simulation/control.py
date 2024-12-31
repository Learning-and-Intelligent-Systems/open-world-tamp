import math
import time

import numpy as np

import owt.pb_utils as pbu


def step_curve(body, joints, curve, time_step=2e-2, print_freq=None, **kwargs):
    start_time = time.time()
    num_steps = 0
    time_elapsed = 0.0
    last_print = time_elapsed
    for num_steps, (time_elapsed, positions) in enumerate(
        pbu.sample_curve(curve, time_step=time_step)
    ):
        pbu.set_joint_positions(body, joints, positions, **kwargs)
        if (print_freq is not None) and (print_freq <= (time_elapsed - last_print)):
            print(
                "Step: {} | Sim secs: {:.3f} | Real secs: {:.3f} | Steps/sec {:.3f}".format(
                    num_steps,
                    time_elapsed,
                    pbu.elapsed_time(start_time),
                    num_steps / pbu.elapsed_time(start_time),
                )
            )
            last_print = time_elapsed
        yield positions
    if print_freq is not None:
        print(
            "Simulated {} steps ({:.3f} sim seconds) in {:.3f} real seconds".format(
                num_steps, time_elapsed, pbu.elapsed_time(start_time)
            )
        )


#######################################################


def simulate_controller(controller, real_per_sim=2.0, print_freq=0.1, hook=None):
    start_time = time.time()
    dt = pbu.get_time_step()
    num_steps = 0
    time_elapsed = 0.0
    last_print = time_elapsed
    pbu.enable_gravity()
    for _ in controller:
        pbu.step_simulation()
        if not (hook is None) and num_steps % 10 == 0:
            hook()
        num_steps += 1
        time_elapsed += dt
        if (print_freq is not None) and (print_freq <= (time_elapsed - last_print)):
            print(
                "Step: {} | Sim secs: {:.3f} | Real secs: {:.3f} | Steps/sec {:.3f}".format(
                    num_steps,
                    time_elapsed,
                    pbu.elapsed_time(start_time),
                    num_steps / pbu.elapsed_time(start_time),
                )
            )
            last_print = time_elapsed
        if pbu.has_gui():
            if real_per_sim is None:
                pbu.wait_if_gui()
            else:
                duration = real_per_sim * dt
                time.sleep(duration)
    if print_freq is not None:
        print(
            "Simulated {} steps ({:.3f} sim seconds) in {:.3f} real seconds".format(
                num_steps, time_elapsed, pbu.elapsed_time(start_time)
            )
        )
    return time_elapsed


#######################################################


def timeout_controller(controller, timeout=np.inf, time_step=None):
    if time_step is None:
        time_step = pbu.get_time_step()
    time_elapsed = 0.0
    for output in controller:
        if time_elapsed > timeout:
            break
        yield output
        time_elapsed += time_step


def stall_for_duration(duration):
    dt = pbu.get_time_step()
    time_elapsed = 0.0
    while time_elapsed < duration:
        yield time_elapsed
        time_elapsed += dt


def hold_for_duration(body, duration, joints=None, **kwargs):
    if joints is None:
        joints = pbu.get_movable_joints(body)
    pbu.control_joints(body, joints, **kwargs)
    return stall_for_duration(duration)


def pose_controller(
    body,
    target_pose,
    pos_tol=1e-3,
    ori_tol=math.radians(1),
    max_force=100 * pbu.GRAVITY,
    timeout=np.inf,
    verbose=False,
):
    # TODO: with statement hold pose
    constraint = pbu.add_pose_constraint(body, pose=target_pose, max_force=max_force)
    target_pos, target_quat = target_pose
    dt = pbu.get_time_step()
    time_elapsed = 0.0
    while time_elapsed < timeout:
        pose = pbu.get_pose(body)
        pos_error = pbu.get_distance(pbu.point_from_pose(pose), target_pos)
        ori_error = pbu.quat_angle_between(pbu.quat_from_pose(pose), target_quat)
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
    pbu.remove_constraint(constraint)


def interpolate_pose_controller(body, target_pose, timeout=np.inf, draw=True, **kwargs):
    pose_waypoints = list(
        pbu.interpolate_poses(
            pbu.get_pose(body),
            target_pose,
            pos_step_size=0.01,
            ori_step_size=math.radians(1),
        )
    )
    dt = pbu.get_time_step()
    time_elapsed = 0.0
    handles = []
    for num, waypoint_pose in enumerate(pose_waypoints):
        if time_elapsed >= timeout:
            break
        is_goal = num == len(pose_waypoints) - 1
        if draw:
            handles.extend(pbu.draw_pose(waypoint_pose, length=0.05))
        for output in pose_controller(body, waypoint_pose, **kwargs):
            yield output
            time_elapsed += dt
        pbu.remove_handles(handles)


def interpolate_controller(
    body,
    joints,
    target_positions,
    max_velocities=None,
    dt=1e-1,
    timeout=np.inf,
    **kwargs
):
    duration_fn = pbu.get_duration_fn(body, joints, velocities=max_velocities)
    positions = pbu.get_joint_positions(body, joints)
    duration = duration_fn(positions, target_positions)
    num_steps = int(np.ceil(duration / dt))
    waypoints = list(pbu.interpolate(positions, target_positions, num_steps))
    time_elapsed = 0.0
    for num, waypoint in enumerate(waypoints):
        if time_elapsed >= timeout:
            break
        for output in pbu.joint_controller(
            body, joints, waypoint, timeout=(timeout - time_elapsed), **kwargs
        ):
            yield output
            time_elapsed += pbu.get_time_step()


#######################################################


def follow_path(
    body,
    joints,
    path,
    waypoint_tol=1e-2 * np.pi,
    goal_tol=5e-3 * np.pi,
    waypoint_timeout=1.0,
    path_timeout=np.inf,
    lead_step=0.1,
    verbose=True,
    **kwargs
):
    start_time = time.time()
    dt = pbu.get_time_step()
    handles = []
    steps = 0
    duration = 0.0
    odometry = [np.array(pbu.get_joint_positions(body, joints))]
    pbu.control_joints(body, pbu.get_movable_joints(body))
    for num, positions in enumerate(path):
        if duration > path_timeout:
            break
        # start = duration
        is_goal = num == len(path) - 1
        tolerance = goal_tol if is_goal else waypoint_tol

        if verbose:
            print(
                "Waypoint: {} | Goal: {} | Sim steps: {} | Sim secs: {:.3f} | Steps/sec {:.3f}".format(
                    num, is_goal, steps, duration, steps / pbu.elapsed_time(start_time)
                )
            )

        if lead_step is None:
            controller = pbu.joint_controller(
                body,
                joints,
                positions,
                tolerance=tolerance,
                timeout=waypoint_timeout,
                **kwargs
            )
        else:
            controller = pbu.waypoint_joint_controller(
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
            steps += 1
            duration += dt
            odometry.append(
                odometry[-1] + dt * np.array(pbu.get_joint_velocities(body, joints))
            )
        pbu.remove_handles(handles)
    if verbose:
        print(
            "Followed {} waypoints in {} sim steps ({:.3f} sim seconds)".format(
                len(path), steps, duration
            )
        )


def follow_curve(body, joints, positions_curve, time_step=1e-1, **kwargs):
    control_times = np.append(
        np.arange(positions_curve.x[0], positions_curve.x[-1], step=time_step),
        [positions_curve.x[-1]],
    )
    path = [positions_curve(control_time) for control_time in control_times]
    return follow_path(body, joints, path, **kwargs)
