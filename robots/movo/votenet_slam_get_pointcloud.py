import sys

# NOTE(caelan): must come before other imports
sys.path.extend(["pddlstream", "pybullet-planning"])

import argparse
import copy
import importlib
# System libs
import os
import pickle
import sys
import time

import numpy as np
# Our libs
import open3d as o3d
import torch
import torch.optim as optim
from movo.movo_utils import (ARMS, COMMAND_MOVO_GROUPS,
                             MOVO_DISABLED_COLLISIONS, MOVO_GROUPS, MOVO_INFOS,
                             MOVO_TOOL_FRAMES, MOVO_URDF, create_floor,
                             gripper_from_arm, side_from_arm)
from open_world.estimation.bounding import estimate_oobb
from open_world.simulation.entities import Manipulator
from pybullet_tools.utils import RGBA, connect, load_pybullet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + "/../vision_utils/votenet"
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "scannet"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "models"))

from ap_helper import parse_predictions
from pc_util import random_sampling, read_ply
from pybullet_tools.utils import (Pose, aabb_from_points, create_mesh, invert,
                                  mesh_from_points, set_pose, tform_points)

from vision_utils.votenet.scannet.scannet_detection_dataset import DC


def preprocess_point_cloud(point_cloud):
    """Prepare the numpy point cloud (N,3) for forward pass."""
    point_cloud = point_cloud[:, 0:3]  # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate(
        [point_cloud, np.expand_dims(height, 1)], 1
    )  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,40000,4)
    return pc


PALETTE = [  # Just some colors
    (152, 223, 138),
    (174, 199, 232),
    (31, 119, 180),
    (255, 187, 120),
    (188, 189, 34),
    (140, 86, 75),
    (255, 152, 150),
    (214, 39, 40),
    (197, 176, 213),
    (148, 103, 189),
    (196, 156, 148),
    (23, 190, 207),
    (247, 182, 210),
    (219, 219, 141),
    (255, 127, 14),
    (227, 119, 194),
    (158, 218, 229),
    (44, 160, 44),
    (112, 128, 144),
    (82, 84, 163),
]


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def point_in_cube(point, corners):
    p1 = corners[0]
    p2 = None
    p3 = None
    p4 = None

    min_dist = float("inf")
    for corner in corners[1:]:
        if dist(p1, corner) < min_dist:
            min_dist = dist(p1, corner)
            p2 = corner

    min_dist = float("inf")
    for corner in corners[1:]:
        v1 = p2 - p1
        v2 = corner - p1
        if np.dot(v1, v2) == 0 and dist(p1, corner) < min_dist:
            min_dist = dist(p1, corner)
            p3 = corner

    for corner in corners[1:]:
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = corner - p1
        if np.dot(v1, v3) == 0 and np.dot(v2, v3) == 0:
            p4 = corner

    i = p2 - p1
    j = p3 - p1
    k = p4 - p1
    v = point - p1

    return (
        np.dot(v, i) > 0
        and np.dot(v, i) < np.dot(i, i)
        and np.dot(v, j) > 0
        and np.dot(v, j) < np.dot(j, j)
        and np.dot(v, k) > 0
        and np.dot(v, k) < np.dot(k, k)
    )


if __name__ == "__main__":
    connect(use_gui=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="scannet",
        help="Dataset: sunrgbd or scannet [default: sunrgbd]",
    )
    parser.add_argument(
        "--num_point", type=int, default=20000, help="Point Number [default: 20000]"
    )
    FLAGS = parser.parse_args()

    # ====== Create floor and robot in pybullet ===========
    floor = create_floor()
    movo_body = load_pybullet(MOVO_URDF, fixed_base=True)

    # Create a robot object using the robot body
    movo_manipulators = {
        side_from_arm(arm): Manipulator(
            arm, gripper_from_arm(arm), MOVO_TOOL_FRAMES[arm]
        )
        for arm in [ARMS[0]]
    }
    robot = Movo_Robot(
        movo_body,
        joint_groups=MOVO_GROUPS,
        command_joint_groups=COMMAND_MOVO_GROUPS,
        manipulators=movo_manipulators,
        ik_info=MOVO_INFOS,
        cameras=[],
        custom_limits={},
        disabled_collisions=MOVO_DISABLED_COLLISIONS,
    )

    LOAD = True
    concave = False

    # Get the image from robot
    if not LOAD:
        PCLFILE = "pcl_temp/{}.pkl".format(str(time.time()))
        pcd_data = robot.read_pointcloud()
        with open(PCLFILE, "wb") as handle:
            pickle.dump(pcd_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        PCLFILE = "pcl_temp/1637278555.3598096.pkl"
        with open(PCLFILE, "rb") as handle:
            pcd_data = pickle.load(handle)

    demo_dir = os.path.join(ROOT_DIR, "demo_files")
    checkpoint_path = os.path.join(demo_dir, "pretrained_votenet_on_scannet.tar")
    pc_path = os.path.join(demo_dir, "input_pc_scannet.ply")

    eval_config_dict = {
        "remove_empty_box": True,
        "use_3d_nms": True,
        "nms_iou": 0.25,
        "use_old_type_nms": False,
        "cls_nms": False,
        "per_class_proposal": False,
        "conf_thresh": 0.2,
        "dataset_config": DC,
    }

    # Init the model and optimzier
    MODEL = importlib.import_module("votenet")  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(
        num_proposal=256,
        input_feature_dim=1,
        vote_factor=1,
        sampling="seed_fps",
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
    ).to(device)
    print("Constructed model.")

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))

    # Load and preprocess input point cloud
    net.eval()  # set model to eval mode (for bn and dp)
    # point_cloud = read_ply(pc_path)
    point_cloud = pcd_data["xyz"]

    pc = preprocess_point_cloud(point_cloud)
    print("Loaded point cloud data: %s" % (pc_path))

    # Model inference
    inputs = {"point_clouds": torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)

    toc = time.time()
    print("Inference time: %f" % (toc - tic))
    end_points["point_clouds"] = inputs["point_clouds"]
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print("Finished detection. %d object detected." % (len(pred_map_cls[0])))
    # print(pred_map_cls[0][0])

    dump_dir = os.path.join(demo_dir, "%s_results" % (FLAGS.dataset))
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    MODEL.dump_results(end_points, dump_dir, DC, True)
    print("Dumped detection results to folder %s" % (dump_dir))

    painted_rgb = copy.deepcopy(pcd_data["rgb"])
    obbs = []
    for boxcls, corners, prob in pred_map_cls[0]:
        corners[:, 1] *= -1
        corners = corners[:, [0, 2, 1]]
        box_corners = o3d.utility.Vector3dVector(corners)
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(box_corners)
        obbs.append(obb)
        total_points = []
        orig_colors = []
        for pi, point in enumerate(pcd_data["xyz"]):
            npcd = o3d.geometry.PointCloud()
            npcd.points = o3d.utility.Vector3dVector([point])
            npcd = npcd.crop(obb)
            if len(npcd.points) > 0:
                orig_colors.append(pcd_data["rgb"][pi])
                # painted_rgb[pi] = PALETTE[boxcls]
                total_points.append(pcd_data["xyz"][pi])

        constrained_oob = estimate_oobb(total_points)
        origin_pose = constrained_oob.pose
        points_origin = tform_points(invert(origin_pose), total_points)
        obj_mesh = (
            concave_mesh(points_origin) if concave else mesh_from_points(points_origin)
        )  # concave_mesh | concave_hull
        VISUALIZE_COLLISION = False
        color = np.mean(np.array(orig_colors), axis=0)
        if (
            PALETTE[boxcls][0] > 128
            and PALETTE[boxcls][1] > 128
            and PALETTE[boxcls][2] < 128
        ):
            color = PALETTE[boxcls]

        color = RGBA(color[0] / 256.0, color[1] / 256.0, color[2] / 256.0, 1)
        if concave:
            obj_estimate = create_concave_mesh(
                obj_mesh, under=False, color=None if VISUALIZE_COLLISION else color
            )
        else:
            obj_estimate = create_mesh(
                obj_mesh, under=True, color=None if VISUALIZE_COLLISION else color
            )
        set_pose(obj_estimate, origin_pose)

    painted_rgb = painted_rgb / 256.0
    # pcd_data['xyz'] = pcd_data['xyz'][:, [0, 2, 1]]
    # pcd_data['xyz'][:, 1] *= -1
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(pcd_data["xyz"])
    out_pcd.colors = o3d.utility.Vector3dVector(painted_rgb)
    # To visualize and save the point cloud with open 3d
    pcd = o3d.io.read_point_cloud("cloud.ply")

    o3d.visualization.draw_geometries([out_pcd] + obbs)
