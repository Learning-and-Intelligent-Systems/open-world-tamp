import math
import os
import time
from itertools import chain

import numpy as np
from pybullet_tools.pr2_utils import (
    TOOL_POSE,
    get_gripper_joints,
    get_group_joints,
    get_side_grasps,
    get_top_grasps,
    gripper_from_side,
)
from pybullet_tools.utils import (
    PI,
    WSG_GRIPPER,
    Euler,
    LockRenderer,
    Point,
    Pose,
    PoseSaver,
    add_data_path,
    add_pose_constraint,
    control_joints_hold,
    draw_point,
    draw_pose,
    dump_body,
    elapsed_time,
    get_joint_positions,
    get_max_limits,
    get_max_velocities,
    get_min_limits,
    get_model_path,
    get_movable_joints,
    get_pose,
    get_pose_distance,
    invert,
    load_pybullet,
    multiply,
    remove_constraint,
    set_joint_positions,
    set_mass,
    set_pose,
    unit_pose,
    wait_if_gui,
)

from open_world.estimation.observation import extract_point_cloud, tform_labeled_points
from open_world.planning.grasping import (
    close_gripper,
    filter_grasps,
    generate_mesh_grasps,
    get_grasp,
    get_pregrasp,
    parallel_gripper_contact,
)
from open_world.planning.graspnet import (
    GRASPNET_POSE,
    load_graspnet_grasps,
    to_graspnet,
)
from open_world.simulation.control import (
    interpolate_pose_controller,
    simulate_controller,
    stall_for_duration,
)
from open_world.simulation.environment import set_grasping_dynamics
from open_world.simulation.lis import SRL_PATH


def simulate_grasp(
    gripper,
    obj,
    grasp_tool,
    gripper_from_tool=unit_pose(),
    gripper_joints=None,
    max_velocities=None,
    **kwargs
):
    if gripper_joints is None:
        gripper_joints = get_movable_joints(gripper)
    if max_velocities is None:
        max_velocities = get_max_velocities(gripper, gripper_joints)

    indices = [i for i, v in enumerate(max_velocities) if v > 0]
    gripper_joints = np.take(gripper_joints, indices)
    max_velocities = np.take(max_velocities, indices)
    closed_conf = np.zeros(len(gripper_joints))

    grasp = get_grasp(grasp_tool, gripper_from_tool)
    pregrasp = get_pregrasp(grasp_tool, gripper_from_tool, **kwargs)

    obj_pose = get_pose(obj)
    grasp_pose = multiply(obj_pose, invert(grasp))
    pregrasp_pose = multiply(obj_pose, invert(pregrasp))
    set_pose(gripper, pregrasp_pose)

    simulate_controller(interpolate_pose_controller(gripper, grasp_pose))

    pose_constraint = add_pose_constraint(gripper)
    simulate_controller(
        close_gripper(
            gripper, gripper_joints, closed_conf, obj, max_velocities=max_velocities
        )
    )
    remove_constraint(pose_constraint)

    simulate_controller(interpolate_pose_controller(gripper, pregrasp_pose))

    # TODO: context saver
    pose_constraint = add_pose_constraint(gripper)
    simulate_controller(stall_for_duration(duration=1.0))
    remove_constraint(pose_constraint)

    # TODO: unify with LTAMP scoring
    simulated_grasp = multiply(invert(get_pose(gripper)), get_pose(obj))
    contact = parallel_gripper_contact(
        gripper, gripper_joints, obj, distance=2e-2
    )  # TODO: report distance
    pos_error, ori_error = get_pose_distance(grasp, simulated_grasp)
    gripper_width = np.average(get_joint_positions(gripper, gripper_joints))
    print(
        "Contact: {} | Width: {:.3f} | Position error: {:.3f} cm | Orientation error {:.3f} degrees".format(
            contact, gripper_width, 100 * pos_error, math.degrees(ori_error)
        )
    )
    # TODO: average over runs

    result = {
        "grasp": grasp_tool,
        "contact": contact,
        "gripper_width": gripper_width,  # TODO: width error
        "position_error": pos_error,
        "orientation_error": ori_error,
    }
    return result


def write_grasps(robot, obj):
    [camera] = robot.cameras
    camera_image = camera.get_image()
    labeled_points = extract_point_cloud(camera_image, bodies=[obj])
    # TODO: create a body from these labeled points

    handles = []
    with LockRenderer():
        for labeled_point in labeled_points:
            handles.extend(draw_point(labeled_point.point, color=labeled_point.color))
    reference_pose = get_pose(obj)
    labeled_points = tform_labeled_points(invert(reference_pose), labeled_points)
    grasps = list(
        get_top_grasps(obj, grasp_length=0.03, under=True, tool_pose=TOOL_POSE)
    )
    to_graspnet(labeled_points, grasps)
    wait_if_gui()
    # TODO: save and send to Aidan


def test_grasp(
    robot,
    side,
    obj,
    obstacles=[],
    clone_gripper=False,
    gripper_name="pr2",
    grasp_method="geometric",
):

    # write_grasps(robot, obj)

    # path = get_mesh_path(obj)
    # name = ycb_type_from_file(os.path.dirname(path))
    if grasp_method == "graspnet":
        grasps_from_name = load_graspnet_grasps(local=False, visualize=False)

    # for grasp, score in randomize(grasps_from_name[name]): # randomize
    #     obj_pose = multiply(tool_pose, GRASPNET_POSE, invert(grasp))
    #     handles = draw_pose(obj_pose)
    #     set_pose(obj, obj_pose)
    #     if not pairwise_collision(robot, obj):
    #         wait_if_gui()
    #     remove_handles(handles)

    # gripper_joints = max_velocities = open_conf = closed_conf = None
    if clone_gripper:
        # gripper = ClonedGripper(robot, link_from_name(robot, PR2_GRIPPER_ROOTS[side]),
        #                        get_group_joints(robot, gripper_from_arm(side)))

        # https://github.com/graspit-simulator/graspit
        _, gripper_group, tool_link = robot.manipulators[side]
        gripper = robot.get_component(gripper_group)
        gripper_joints = robot.get_component_joints(gripper_group)
        tool_from_root = invert(robot.get_parent_from_tool(side))

        # gripper_from_robot = robot.get_component_mapping(gripper_group)
        # open_conf = robot.get_component_info(get_max_limits, gripper_group) # TODO: use this instead
        robot_gripper_joints = get_group_joints(robot, gripper_from_side(side))
        open_conf = get_max_limits(robot, robot_gripper_joints)
        get_min_limits(robot, robot_gripper_joints)
        max_velocities = get_max_velocities(robot, robot_gripper_joints)
        max_width = robot.get_max_gripper_width(get_gripper_joints(robot, side))
    else:
        if gripper_name == "pr2":
            gripper_path = "models/ltamp/pr2_description/pr2_l_gripper.urdf"
            tool_from_root = Pose(point=Point(x=-0.12), euler=Euler(pitch=0))
        elif gripper_name == "wsg_50":
            gripper_path = get_model_path(
                "models/drake/wsg_50_description/urdf/wsg_50_mesh_visual.urdf"
            )
            tool_from_root = Pose(
                point=Point(x=-0.12), euler=Euler(roll=PI / 2, yaw=PI / 2)
            )
        elif gripper_name == "panda":
            gripper_path = os.path.join(
                SRL_PATH, "franka_description", "robots", "hand.urdf"
            )
            tool_from_root = Pose(point=Point(x=-0.12), euler=Euler(pitch=PI / 2))
        elif gripper_name == "pybullet":
            add_data_path()
            gripper_path = WSG_GRIPPER  # PR2_GRIPPER | WSG_GRIPPER
            # tool_from_root = Pose(point=Point(x=-0.28)) # PR2_GRIPPER
            tool_from_root = Pose(
                point=Point(x=-0.28), euler=Euler(pitch=PI / 2)
            )  # TODO: WSG_GRIPPER
        else:
            raise NotImplementedError(gripper_name)
        gripper = load_pybullet(gripper_path)
        if isinstance(gripper, tuple):  # *.sdf
            [gripper] = gripper
        # tool_from_root = Pose(Point(z=-0.1), Euler(yaw=3*PI/4)) # right_gripper
        # draw_pose(get_link_pose(gripper, link_from_name(gripper, 'left_gripper')))
        if gripper_name == "wsg_50":
            set_mass(gripper, 1e-3)  # world link

        gripper_joints = get_movable_joints(gripper)
        open_conf = get_max_limits(gripper, gripper_joints)
        get_min_limits(gripper, gripper_joints)
        max_velocities = None
        max_width = robot.get_max_gripper_width(gripper_joints)

    draw_pose(invert(tool_from_root), parent=gripper)
    # draw_pose(Pose(), parent=gripper)
    dump_body(gripper)
    print("Gripper width: {:.3f}".format(max_width))

    for link in robot.get_finger_links(gripper_joints):
        set_grasping_dynamics(gripper, link)  # TODO: dynamics aren't cloned

    # TODO: separate out and join with streams.py
    if grasp_method == "simple":
        grasp_candidates = chain(
            get_top_grasps(obj, grasp_length=0.03, under=True, tool_pose=TOOL_POSE),
            get_side_grasps(
                obj, grasp_length=0.03, under=False, tool_pose=TOOL_POSE
            ),  # top_offset=0., # TODO: top offset is off for under=True
        )
        # Attempts: 4 | Success rate: 1.000 | Elapsed time: 3.525
    elif grasp_method == "graspnet":
        name = "potted_meat_can"
        print("Name: {} | Grasps: {}".format(name, len(grasps_from_name[name])))
        # TODO: apply GRASPNET_POSE elsewhere
        grasp_candidates = [
            multiply(GRASPNET_POSE, invert(grasp))
            for grasp, _ in grasps_from_name[name]
        ]
        # grasp_candidates = randomize(grasp_candidates)
        # Success rate: 0.160 | Elapsed time: 495.301
    elif grasp_method == "geometric":
        # pitches = [0] # Top
        # pitches = [-PI/2, PI/2] # Side
        pitches = [-PI / 2, 0, PI / 2]  # Both
        grasp_candidates = generate_mesh_grasps(
            obj, pitches=pitches, discrete_pitch=False, max_width=max_width
        )
        # Attempts: 1000 | Success rate: 1.000 | Elapsed time: 1212.090
    else:
        raise NotImplementedError(grasp_method)

    # import pybullet as p
    # synchronize the visualizer (rendering frames for the video mp4) with stepSimulation
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/examples/video_sync_mp4.py#L28
    # p.COV_ENABLE_SINGLE_STEP_RENDERING

    # with VideoSaver('video.mp4'):
    start_time = time.time()
    results = []
    control_joints_hold(robot, get_movable_joints(robot))
    set_joint_positions(gripper, gripper_joints, open_conf)
    for grasp_tool in filter_grasps(
        gripper,
        obj,
        grasp_candidates,
        invert(tool_from_root),
        obstacles=obstacles,
        draw=False,
    ):
        with PoseSaver(obj):
            set_joint_positions(gripper, gripper_joints, open_conf)
            result = simulate_grasp(
                gripper,
                obj,
                grasp_tool,
                gripper_from_tool=invert(tool_from_root),
                gripper_joints=gripper_joints,
                max_velocities=max_velocities,
            )
            results.append(result)
            wait_if_gui()
            set_joint_positions(gripper, gripper_joints, open_conf)
            control_joints_hold(gripper, gripper_joints)
            if len(results) >= 1000:
                break

    # TODO: analyze the volume of the space that is covered by a successful grasp
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors

    # TODO: compare grasp quality and success rate
    success_rate = float(sum(result["contact"] for result in results)) / len(results)
    print(
        "Attempts: {} | Success rate: {:.3f} | Elapsed time: {:.3f}".format(
            len(results), success_rate, elapsed_time(start_time)
        )
    )
    wait_if_gui()
    # test_pick(gripper, obj, grasp_tool, invert(tool_from_root), gripper_joints, max_velocities)
