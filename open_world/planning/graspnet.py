import math
import os
import pickle
import time
from collections import defaultdict

import numpy as np
from pybullet_tools.pr2_utils import TOOL_POSE
from pybullet_tools.utils import (
    MAX_RGB,
    PI,
    WHITE,
    Euler,
    Point,
    Pose,
    apply_alpha,
    control_joints_hold,
    create_mesh,
    create_obj,
    draw_pose,
    get_link_pose,
    get_links,
    get_max_limits,
    get_movable_joints,
    get_pose_distance,
    has_gui,
    invert,
    load_pybullet,
    mesh_from_points,
    multiply,
    pairwise_collision,
    pose_from_tform,
    read_pickle,
    remove_body,
    remove_handles,
    safe_zip,
    set_joint_positions,
    set_pose,
    tform_from_pose,
    wait_if_gui,
)

from open_world.planning.grasping import get_grasp
from open_world.planning.primitives import Grasp
from open_world.simulation.lis import GRASPNET_DIR, USING_ROS, get_ycb_obj_path

GRASPNET_POSE = multiply(
    TOOL_POSE,  # Pose(euler=Euler(pitch=PI/2)),
    Pose(euler=Euler(yaw=PI / 2)),  # +/-
    Pose(point=Point(z=-0.1)),
)


def visualize_graspnet_grasps(robot, obj, grasp_candidates):
    gripper_path = "models/ltamp/pr2_description/pr2_l_gripper.urdf"

    gripper = load_pybullet(gripper_path)
    if isinstance(gripper, tuple):  # *.sdf
        [gripper] = gripper
    tool_from_root = Pose(point=Point(x=-0.12), euler=Euler(pitch=0))

    gripper_joints = get_movable_joints(gripper)
    open_conf = get_max_limits(gripper, gripper_joints)

    control_joints_hold(robot, get_movable_joints(robot))
    set_joint_positions(gripper, gripper_joints, open_conf)
    for gi, grasp_candidate in enumerate(grasp_candidates):
        gripper_from_tool = invert(tool_from_root)
        obj_pose = obj.observed_pose
        gripper = load_pybullet(gripper_path)
        grasp = get_grasp(grasp_candidate, gripper_from_tool)
        grasp_pose = multiply(obj_pose, invert(grasp))

        set_joint_positions(gripper, gripper_joints, open_conf)
        set_pose(gripper, grasp_pose)

        if any(pairwise_collision(gripper, obst) for obst in [obj]):
            remove_body(gripper)
            continue

        gripper_links = get_links(gripper)
        base_link_pose = get_link_pose(gripper, gripper_links[0])[0][2]
        tip_link_pose = (
            get_link_pose(gripper, gripper_links[6])[0][2]
            + get_link_pose(gripper, gripper_links[8])[0][2]
        ) / 2.0
        base_link_pose > tip_link_pose

        # else:
        #     remove_body(gripper)
        wait_if_gui()

    while True:
        time.sleep(0.1)


def get_graspnet_gen_fn(
    robot,
    grasp_dataset_filename="./temp_grasps/grasp_data.pkl",
    load_grasps=True,
    save_grasps=False,
    visualize=False,
    **kwargs
):
    import pybullet
    from scipy.spatial.transform import Rotation

    side = robot.get_arbitrary_side()
    arm_group, gripper_group, tool_name = robot.manipulators[side]
    closed_conf, open_conf = robot.get_group_limits(gripper_group)

    robot.set_group_positions(gripper_group, open_conf)

    robot.get_component(gripper_group)
    robot.get_parent_from_tool(side)

    def gen_fn(obj):
        pc = np.array(
            [[p.point[1], p.point[2], p.point[0]] for p in obj.labeled_points]
        )
        pc_color = np.array([p.color for p in obj.labeled_points])[:, :3] * 255

        # Pass through graspnet to get poses
        if load_grasps:
            with open(grasp_dataset_filename, "rb") as h:
                grasp_dataset = pickle.load(h)
                grasp_outputs, grasp_scores = (
                    grasp_dataset["grasp_outputs"],
                    grasp_dataset["grasp_scores"],
                )

        else:
            # Pass through graspnet to get poses
            if USING_ROS:
                from grasp.utils import query_grasp_server

                grasp_outputs, grasp_scores = query_grasp_server(
                    pc, grasp_mode="graspnet"
                )
                grasp_outputs = list(map(tform_from_pose, grasp_outputs))
            else:
                from grasp.graspnet_interface import generate_grasps

                grasp_outputs, grasp_scores = generate_grasps(pc, pc_color)

        if save_grasps:
            with open(grasp_dataset_filename, "wb") as h:
                pickle.dump(
                    {"grasp_outputs": grasp_outputs, "grasp_scores": grasp_scores}, h
                )

        # scored_grasps = safe_zip(grasp_outputs, grasp_scores)
        # sorted_pairs = sorted(scored_grasps, key=lambda pair: pair[1], reverse=True)
        # sorted_grasp_outputs = [sorted_pair[0] for sorted_pair in sorted_pairs]

        grasp_candidates = []
        for grasp in grasp_outputs:
            p1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            gripper_position_vec = grasp[:3, 3]
            gripper_position_vec = [
                gripper_position_vec[2],
                gripper_position_vec[0],
                gripper_position_vec[1],
            ]
            gr = Rotation.from_matrix(np.matmul(p1, grasp[:3, :3]))
            groll, gpitch, gyaw = list(gr.as_euler("xyz", degrees=False))
            gquat = pybullet.getQuaternionFromEuler([groll, gpitch, gyaw])
            new_grasp = (tuple(gripper_position_vec), gquat)
            grasp_candidates.append(new_grasp)

        pose = obj.observed_pose
        grasp_candidates = [multiply(invert(pose), grasp) for grasp in grasp_candidates]
        grasp_candidates = [
            multiply(GRASPNET_POSE, invert(grasp)) for grasp in grasp_candidates
        ]

        if visualize:
            visualize_graspnet_grasps(robot, obj, grasp_candidates)

        for gpose in grasp_candidates:
            yield (Grasp(obj, gpose, closed_position=closed_conf),)

    return gen_fn


##################################################


def to_graspnet(labeled_points, grasps):
    # TODO: theta
    points = [labeled_point.point for labeled_point in labeled_points]
    colors = [MAX_RGB * labeled_point.color[:3] for labeled_point in labeled_points]
    grasps = list(map(tform_from_pose, grasps))
    return {
        "pc": points,
        "pc_colors": colors,
        "grasps": grasps,
        #'grasp_scores': None,
    }


def filter_identical_grasps(scored_grasps):
    # TODO: filter unordered identical grasp predictions
    scored_grasps = sorted(scored_grasps, key=lambda pair: pair[1], reverse=True)
    selected_grasps = scored_grasps[:1]
    for grasp2, score2 in scored_grasps[1:]:
        grasp1, _ = selected_grasps[-1]
        pos_dist, ori_dis = get_pose_distance(grasp1, grasp2)
        if (pos_dist > 1e-6) or (ori_dis > math.radians(1e-1)):
            selected_grasps.append((grasp2, score2))
    return selected_grasps


def load_graspnet_grasps(local=False, visualize=False):
    # https://github.com/rdiankov/openrave/blob/master/python/databases/grasping.py

    grasps_from_name = defaultdict(list)
    for grasp_file in sorted(os.listdir(GRASPNET_DIR))[
        ::-1
    ]:  # TODO: method that returns paths
        name, _ = os.path.splitext(grasp_file)
        # name = grasp_file[:grasp_file.rindex('_')]
        # _, name, _ = grasp_file.split('_')
        theta = float(name.split("_")[-1])
        name = "_".join(name.split("_")[1:-1])
        pose = Pose(Point(y=0.735), Euler(roll=-PI / 2, pitch=theta - PI / 2))
        # if name not in ['cracker_box']:
        #     continue

        grasp_path = os.path.join(GRASPNET_DIR, grasp_file)
        data = read_pickle(grasp_path)
        # print(data.keys()) # dict_keys(['pc', 'pc_colors', 'grasps', 'grasp_scores'])
        point_cloud = data["pc"]
        point_colors = data["pc_colors"]
        assert len(point_cloud) == len(point_colors)
        color = np.mean(point_colors, axis=0) / MAX_RGB

        # handles = []
        # with LockRenderer():
        #     for point, color in random.sample(safe_zip(point_cloud, point_colors), 1000):
        #         handles.extend(draw_point(point, color=apply_alpha(color / MAX_RGB)))
        # wait_if_gui()

        grasps = list(map(pose_from_tform, data["grasps"]))
        grasp_scores = data["grasp_scores"]
        assert len(grasps) == len(grasp_scores)
        if not local:
            grasps = [multiply(invert(pose), grasp) for grasp in grasps]

        # TODO: apply GRASPNET_POSE here
        # grasps = [multiply(GRASPNET_POSE, invert(grasp)) for grasp in grasps]
        scored_grasps = safe_zip(grasps, grasp_scores)
        # scored_grasps = randomize(scored_grasps)
        scored_grasps = sorted(scored_grasps, key=lambda pair: pair[1], reverse=True)

        scored_grasps = filter_identical_grasps(scored_grasps)
        grasps_from_name[name].extend(scored_grasps)

        # File: {} | # grasp_file
        print(
            "Name: {} | Theta: {:.3f} | Cloud: {} | Grasps: {} | Min: {:.3f} | Max: {:.3f}".format(
                name,
                theta,
                len(point_cloud),
                len(grasps),
                min(grasp_scores),
                max(grasp_scores),
            )
        )
        if not has_gui() or not visualize:
            continue

        mesh = mesh_from_points(point_cloud)
        obj1 = create_mesh(mesh, under=True, color=apply_alpha(color, alpha=1))
        obj2 = create_obj(get_ycb_obj_path(name), color=WHITE)  # , **kwargs)
        if local:
            set_pose(obj2, pose)
        else:
            set_pose(obj1, invert(pose))

        for i, (grasp, score) in enumerate(scored_grasps):
            print(
                "Name: {} | Theta: {:.3f} | Index: {} | Score: {}".format(
                    name, theta, i, score
                )
            )
            grasp_pose = grasp
            # grasp_pose = multiply(get_pose(obj1), invert(grasp))
            handles = draw_pose(grasp_pose)
            wait_if_gui()
            remove_handles(handles)

        # wait_if_gui()
        remove_body(obj1)
        remove_body(obj2)

    for name, scored_grasps in grasps_from_name.items():
        grasps_from_name[name] = sorted(
            scored_grasps, key=lambda pair: pair[1], reverse=True
        )
        print(
            "Name: {} | Grasps: {} | Best: {:.3f} | Worst: {:.3f}".format(
                name,
                len(grasps_from_name[name]),
                grasps_from_name[name][0][1],
                grasps_from_name[name][-1][1],
            )
        )
    return grasps_from_name
