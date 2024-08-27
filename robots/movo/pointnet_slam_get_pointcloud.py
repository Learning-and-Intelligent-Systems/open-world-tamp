import sys

# NOTE(caelan): must come before other imports
sys.path.extend(["pddlstream", "pybullet-planning"])

import csv
import importlib
# System libs
import os

import numpy as np
# Our libs
import open3d as o3d
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from movo.movo_utils import (ARMS, BASE_LINK, COMMAND_MOVO_GROUPS,
                             MOVO_CLOSED_CONF, MOVO_DISABLED_COLLISIONS,
                             MOVO_GROUPS, MOVO_INFOS, MOVO_OPEN_CONF,
                             MOVO_TOOL_FRAMES, MOVO_URDF, arm_from_side,
                             create_floor, get_default_conf,
                             get_full_default_conf, gripper_from_arm,
                             side_from_arm)
from movo.run_movo import Movo_Robot
from open_world.estimation.dnn import init_seg, str_from_int_seg_general
from open_world.estimation.geometry import cloud_from_depth
from open_world.estimation.observation import (image_from_labeled,
                                               save_camera_images)
from open_world.estimation.tables import estimate_surfaces
from open_world.planning.planner import (DEFAULT_SHAPE, PARAM, And, Exists,
                                         ForAll, Imply, On)
from open_world.planning.primitives import GroupConf
from open_world.planning.streams import get_plan_motion_fn
from open_world.real_world.movo_controller import MovoController
from open_world.simulation.entities import BOWL, Manipulator, Robot
from open_world.simulation.lis import YCB_MASSES
from open_world.simulation.policy import Policy, run_policy
from open_world.simulation.tasks import Task
from plyfile import PlyData, PlyElement
from pybullet_tools.utils import connect, load_pybullet

from run_planner import create_parser
from vision_utils.pointnet_scene_seg.lib.enet import create_enet_for_3d

PALETTE = [
    (152, 223, 138),  # floor
    (174, 199, 232),  # wall
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (247, 182, 210),  # desk
    (219, 219, 141),  # curtain
    (255, 127, 14),  # refrigerator
    (227, 119, 194),  # bathtub
    (158, 218, 229),  # shower curtain
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (82, 84, 163),  # otherfurn
]

NYUCLASSES = [
    "floor",
    "wall",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "bathtub",
    "shower curtain",
    "toilet",
    "sink",
    "otherprop",
]
NUM_CLASSES = len(NYUCLASSES)


def visualize(args, preds):
    vertex = []
    for i in range(preds.shape[0]):
        vertex.append(
            (
                preds[i][0],
                preds[i][1],
                preds[i][2],
                preds[i][3],
                preds[i][4],
                preds[i][5],
            )
        )

    vertex = np.array(
        vertex,
        dtype=[
            ("x", np.dtype("float32")),
            ("y", np.dtype("float32")),
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8")),
        ],
    )

    output_pc = PlyElement.describe(vertex, "vertex")
    output_pc = PlyData([output_pc])
    output_root = "./temp"
    os.makedirs(output_root, exist_ok=True)
    output_pc.write(os.path.join(output_root, "scene0000_00.ply"))


def filter_points(coords, preds):
    assert coords.shape[0] == preds.shape[0]

    _, coord_ids = np.unique(coords, axis=0, return_index=True)
    coord_filtered, pred_filtered = coords[coord_ids], preds[coord_ids]
    # coord_filtered, pred_filtered = coords, preds
    filtered = []
    for point_idx in range(coord_filtered.shape[0]):
        filtered.append(
            [
                coord_filtered[point_idx][0],
                coord_filtered[point_idx][1],
                coord_filtered[point_idx][2],
                PALETTE[pred_filtered[point_idx]][0],
                PALETTE[pred_filtered[point_idx]][1],
                PALETTE[pred_filtered[point_idx]][2],
            ]
        )

    return np.array(filtered)


def forward(args, model, coords, feats):
    pred = []
    coord_chunk, feat_chunk = torch.split(
        coords.squeeze(0), args.batch_size, 0
    ), torch.split(feats.squeeze(0), args.batch_size, 0)
    assert len(coord_chunk) == len(feat_chunk)
    for coord, feat in zip(coord_chunk, feat_chunk):
        output = model(torch.cat([coord, feat], dim=2))
        pred.append(output)

    pred = torch.cat(pred, dim=0)  # (CK, N, C)
    outputs = pred.max(2)[1]

    return outputs


def predict_label(args, model, coords, feats):

    output_coords, output_preds = [], []
    preds = forward(args, model, coords, feats)

    # dump
    coords = coords.squeeze(0).view(-1, 3).cpu().numpy()
    preds = preds.view(-1).cpu().numpy()
    output_coords.append(coords)
    output_preds.append(preds)

    print("filtering points...")
    output_coords = np.concatenate(output_coords, axis=0)
    output_preds = np.concatenate(output_preds, axis=0)
    filtered = filter_points(output_coords, output_preds)

    return filtered


def get_point_sets(args):
    # Load in scene points
    scene_data = np.load(
        "/home/aidan/open-world-tamp/vision_utils/pointnet_scene_seg/preprocessing/scannet_scenes/scene0000_00.npy"
    )
    label = scene_data[:, 10].astype(np.int32)
    npoints = 8192

    # unpack
    point_set_ini = scene_data[:, :3]  # include xyz by default
    color = scene_data[:, 3:6] / 255.0  # normalize the rgb values to [0, 1]
    normal = scene_data[:, 6:9]

    if args.use_color:
        point_set_ini = np.concatenate([point_set_ini, color], axis=1)

    if args.use_normal:
        point_set_ini = np.concatenate([point_set_ini, normal], axis=1)

    semantic_seg_ini = label.astype(np.int32)
    coordmax = point_set_ini[:, :3].max(axis=0)
    coordmin = point_set_ini[:, :3].min(axis=0)
    xlength = 1.5
    ylength = 1.5
    nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / xlength).astype(np.int32)
    nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / ylength).astype(np.int32)
    point_sets = list()
    semantic_segs = list()
    sample_weights = list()

    for i in range(nsubvolume_x):
        for j in range(nsubvolume_y):
            curmin = coordmin + [i * xlength, j * ylength, 0]
            curmax = coordmin + [
                (i + 1) * xlength,
                (j + 1) * ylength,
                coordmax[2] - coordmin[2],
            ]
            mask = (
                np.sum(
                    (point_set_ini[:, :3] >= (curmin - 0.01))
                    * (point_set_ini[:, :3] <= (curmax + 0.01)),
                    axis=1,
                )
                == 3
            )
            cur_point_set = point_set_ini[mask, :]
            cur_semantic_seg = semantic_seg_ini[mask]
            if len(cur_semantic_seg) == 0:
                continue

            choice = np.random.choice(len(cur_semantic_seg), npoints, replace=True)
            point_set = cur_point_set[choice, :]  # Nx3
            point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3

    point_sets = np.concatenate(tuple(point_sets), axis=0)

    return point_sets


def to_tensor(arr):
    return torch.Tensor(arr).cuda()


##################### compute_multiview_features (preproc step1) #####################


def _resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims != new_image_dims:
        resize_width = int(
            math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1]))
        )
        image = transforms.Resize(
            [new_image_dims[1], resize_width], interpolation=Image.NEAREST
        )(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    return np.array(image)


def create_enet():
    enet_fixed, enet_trainable, _ = create_enet_for_3d(41, ENET_PATH, 21)
    enet = nn.Sequential(enet_fixed, enet_trainable).cuda()
    enet.eval()
    for param in enet.parameters():
        param.requires_grad = False

    return enet


def compute_multiview_features(image):
    new_image_dims = [328, 256]
    image = _resize_crop_image(image, image_dims)
    if len(image.shape) == 3:  # color image
        image = np.transpose(image, [2, 0, 1])  # move feature to front
        image = transforms.Normalize(
            mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129]
        )(torch.Tensor(image.astype(np.float32) / 255.0))
    elif len(image.shape) == 2:  # label image
        image = np.expand_dims(image, 0)
    else:
        raise ValueError
    enet = create_enet()
    features = enet(images)
    return features


##################### compute_multiview_projection (preproc step 2) #####################
def compute_projection_s2(points, depth, camera_to_world):
    """
    :param points: tensor containing all points of the point cloud (num_points, 3)
    :param depth: depth map (size: proj_image)
    :param camera_to_world: camera pose (4, 4)

    :return indices_3d (array with point indices that correspond to a pixel),
    :return indices_2d (array with pixel indices that correspond to a point)

    note:
        the first digit of indices represents the number of relevant points
        the rest digits are for the projection mapping
    """
    num_points = points.shape[0]
    num_frames = depth.shape[0]
    indices_3ds = torch.zeros(num_frames, num_points + 1).long().cuda()
    indices_2ds = torch.zeros(num_frames, num_points + 1).long().cuda()

    for i in range(num_frames):
        indices = PROJECTOR.compute_projection(
            to_tensor(points), to_tensor(depth[i]), to_tensor(camera_to_world[i])
        )
        if indices:
            indices_3ds[i] = indices[0].long()
            indices_2ds[i] = indices[1].long()

    return indices_3ds, indices_2ds


def compute_multiview_projection(point_cloud, frame_list):

    # load frames
    scene_images = np.zeros((len(frame_list), 3, 256, 328))
    scene_depths = np.zeros((len(frame_list), 32, 41))
    scene_poses = np.zeros((len(frame_list), 4, 4))
    for i, frame_id in enumerate(frame_list):
        scene_images[i], scene_depths[i], scene_poses[i] = frame_id

    projection_3d, projection_2d = compute_projection_s2(
        point_cloud[:, :3], scene_depths, scene_poses
    )


##################### project_multiview_features (preproc step 3) #####################


def compute_projection_s3(points, depth, camera_to_world):
    """
    :param points: tensor containing all points of the point cloud (num_points, 3)
    :param depth: depth map (size: proj_image)
    :param camera_to_world: camera pose (4, 4)

    :return indices_3d (array with point indices that correspond to a pixel),
    :return indices_2d (array with pixel indices that correspond to a point)

    note:
        the first digit of indices represents the number of relevant points
        the rest digits are for the projection mapping
    """
    num_points = points.shape[0]
    num_frames = depth.shape[0]
    indices_3ds = torch.zeros(num_frames, num_points + 1).long().cuda()
    indices_2ds = torch.zeros(num_frames, num_points + 1).long().cuda()

    for i in range(num_frames):
        indices = PROJECTOR.compute_projection(
            to_tensor(points), to_tensor(depth[i]), to_tensor(camera_to_world[i])
        )
        if indices:
            indices_3ds[i] = indices[0].long()
            indices_2ds[i] = indices[1].long()

    return indices_3ds, indices_2ds


def project_multiview_features(pointcloud, frame_list):
    scene_images = np.zeros((len(frame_list), 3, 256, 328))
    scene_depths = np.zeros((len(frame_list), 32, 41))
    scene_poses = np.zeros((len(frame_list), 4, 4))
    for i, frame_id in enumerate(frame_list):
        scene_images[i], scene_depths[i], scene_poses[i] = frame_id

    # compute projections for each chunk
    projection_3d, projection_2d = compute_projection_s3(
        pointcloud, scene_depths, scene_poses
    )
    _, inds = torch.sort(projection_3d[:, 0], descending=True)
    projection_3d, projection_2d = projection_3d[inds], projection_2d[inds]

    # compute valid projections
    projections = []
    for i in range(projection_3d.shape[0]):
        num_valid = projection_3d[i, 0]
        if num_valid == 0:
            continue

        projections.append(
            (frame_list[inds[i].long().item()], projection_3d[i], projection_2d[i])
        )

    # project
    point_features = to_tensor(scene).new(scene.shape[0], 128).fill_(0)
    for i, projection in enumerate(projections):
        frame_id = projection[0]
        projection_3d = projection[1]
        projection_2d = projection[2]
        feat = to_tensor(np.load(ENET_FEATURE_PATH.format(scene_id, frame_id)))
        proj_feat = PROJECTOR.project(
            feat, projection_3d, projection_2d, scene.shape[0]
        ).transpose(1, 0)
        if i == 0:
            point_features = proj_feat
        else:
            mask = ((point_features == 0).sum(1) == 128).nonzero().squeeze(1)
            point_features[mask] = proj_feat[mask]


########################################################################################

if __name__ == "__main__":
    connect(use_gui=True)
    np.set_printoptions(
        precision=3, threshold=3, suppress=True
    )  # , edgeitems=1) #, linewidth=1000)
    parser = create_parser()
    parser.add_argument(
        "--batch_size", type=int, help="size of the batch/chunk", default=1
    )
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument(
        "--no_bn",
        action="store_true",
        help="do not apply batch normalization in pointnet++",
    )
    parser.add_argument(
        "--no_xyz",
        action="store_true",
        help="do not apply coordinates as features in pointnet++",
    )
    parser.add_argument(
        "--use_msg", action="store_true", help="apply multiscale grouping or not"
    )
    parser.add_argument(
        "--use_color", action="store_false", help="use color values or not"
    )
    parser.add_argument("--use_normal", action="store_false", help="use normals or not")
    parser.add_argument(
        "--use_multiview",
        action="store_true",
        help="use multiview image features or not",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

    # Get the image from robot
    pcd_data = robot.read_pointcloud()
    out_pcd = o3d.geometry.PointCloud()

    print(pcd_data["xyz"].shape)
    out_pcd.points = o3d.utility.Vector3dVector(pcd_data["xyz"])
    out_pcd.colors = o3d.utility.Vector3dVector(pcd_data["rgb"] / 256)

    # To visualize and save the point cloud with open 3d
    pcd = o3d.io.read_point_cloud("cloud.ply")
    o3d.visualization.draw_geometries([out_pcd])

    # prepare data
    print("preparing data...")
    scene_list = ["scene0000_00"]

    # load model
    print("loading model...")
    model_path = "/home/aidan/open-world-tamp/vision_utils/pointnet_scene_seg/outputs/temp/model.pth"
    Pointnet = importlib.import_module(
        "vision_utils.pointnet_scene_seg.pointnet2.pointnet2_semseg"
    )
    input_channels = (
        int(args.use_color) * 3
        + int(args.use_normal) * 3
        + int(args.use_multiview) * 128
    )
    model = Pointnet.get_model(
        num_classes=NUM_CLASSES,
        is_msg=args.use_msg,
        input_channels=input_channels,
        use_xyz=not args.no_xyz,
        bn=not args.no_bn,
    ).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # predict
    print("predicting...")
    point_sets = get_point_sets(args)
    point_sets = torch.unsqueeze(torch.FloatTensor(point_sets), dim=0).cuda()

    coords = point_sets[:, :, :, :3]
    feats = point_sets[:, :, :, 3:]

    preds = predict_label(args, model, coords, feats)

    # visualize
    print("visualizing...")
    visualize(args, preds)
