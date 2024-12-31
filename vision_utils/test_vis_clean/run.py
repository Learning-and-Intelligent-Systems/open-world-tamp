#!/usr/bin/env python

from __future__ import print_function

try:
    import pybullet as p
except ImportError:
    raise ImportError(
        "This example requires PyBullet (https://pypi.org/project/pybullet/)"
    )

import argparse
import cProfile
import pstats
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from .constant import MASKRCNN_DIR, POSE_DIR, UOIS_DIR, YCB_BANK_DIR

sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
        #'pddlstream/examples/pybullet/utils',
        POSE_DIR,
        UOIS_DIR,
    ]
)
import copy
import os

import cv2
import torch
import torchvision.transforms as transforms
from pybullet_tools.pr2_problems import create_floor
from pybullet_tools.utils import (LockRenderer, WorldSaver, add_data_path,
                                  connect, create_box, disable_gravity,
                                  disconnect, enable_gravity, get_pose,
                                  save_image, set_client, set_euler,
                                  set_joint_positions, set_point, set_pose,
                                  set_quat, step_simulation, wait_for_user)

from .primitives import get_image
from .utils import ICC

BASE_CONSTANT = 1
BASE_VELOCITY = 0.5
EMPTY = -1
BLOCKED = -2
UNOBSERVED = -3

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [
    -1,
    40,
    80,
    120,
    160,
    200,
    240,
    280,
    320,
    360,
    400,
    440,
    480,
    520,
    560,
    600,
    640,
    680,
]
num_points = 1000
img_width = 480
img_length = 640
num_obj = 21
bs = 1


def init_vision_utils(maskrcnn=False, uois=False, densefusion=False):
    """Detection & Segmentation."""
    if maskrcnn:
        import detectron2
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer

        from .constant import MASKRCNN_CONFIDENCE_THRESHOLD

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 23
        cfg.MODEL.WEIGHTS = os.path.join(MASKRCNN_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            MASKRCNN_CONFIDENCE_THRESHOLD  # set the testing threshold for this model
        )
        mask_predictor = DefaultPredictor(cfg)
    elif uois:
        import src.segmentation as uois_segmentation

        from .constant import dsn_config, rrn_config, uois3d_config

        checkpoint_dir = os.path.join(UOIS_DIR, "checkpoints")
        dsn_filename = os.path.join(
            checkpoint_dir, "DepthSeedingNetwork_3D_TOD_checkpoint.pth"
        )
        rrn_filename = os.path.join(checkpoint_dir, "RRN_OID_checkpoint.pth")
        uois3d_config["final_close_morphology"] = "TableTop_v5" in rrn_filename
        mask_predictor = uois_segmentation.UOISNet3D(
            uois3d_config, dsn_filename, dsn_config, rrn_filename, rrn_config
        )
    else:
        mask_predictor = None

    """ 6Dof Pose """
    if densefusion:
        from lib.network import PoseNet, PoseRefineNet

        estimator = PoseNet(num_points=num_points, num_obj=num_obj)
        estimator.cuda()
        estimator.load_state_dict(
            torch.load(
                f"{POSE_DIR}/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth"
            )
        )
        estimator.eval()

        refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
        refiner.cuda()
        refiner.load_state_dict(
            torch.load(
                f"{POSE_DIR}/trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth"
            )
        )
        refiner.eval()

    toolbox = {}
    toolbox["mask"] = mask_predictor
    if densefusion:
        toolbox["pose_estimator"] = estimator
        toolbox["pose_refiner"] = refiner

    return toolbox


def get_bbox(idx, segment):
    py, px = np.where(segment == idx)
    rmin, rmax, cmin, cmax = np.min(py), np.max(py), np.min(px), np.max(px)

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def get_pose6d(
    rgba, depth, xyz, segment, estimator, view_matrix_real, proj_matrix_real, id2class
):
    """
    input:
        rgba: HxWx4.
        depth: HxW. NOTE: This depth is only used for checking the holes, the
        cloud is still from xyz. some changes will be needed if want to applied
        on real images

        xyz: HxWx3. NOTE: The coordinate is different. The positive direction
        of z axis is inversed. Also, the coordinate is different with pybullet
        default coordinate, so be careful.

        segment: HxW.

        id2class: dict. maps id in `segment` to class label.
    """
    from lib.transformations import quaternion_from_matrix, quaternion_matrix

    xyz[:, :, 2] *= -1
    xyz[:, :, 1] *= -1
    obj_list = np.unique(segment)
    pred_wo_refine = {}
    # estimator, refiner = estimator
    for obj in obj_list:
        class_id = id2class[obj]
        if class_id <= 1:  # skip floor and desk
            continue
        rmin, rmax, cmin, cmax = get_bbox(obj, segment)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(segment, obj))
        mask = mask_label * mask_depth
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), "wrap")

        cloud = xyz[rmin:rmax, cmin:cmax, :].reshape(-1, 3)[choose].astype(np.float32)
        cloud = torch.as_tensor(cloud).cuda().view(1, num_points, 3)

        img_masked = rgba[:, :, :3].transpose(2, 0, 1)[:, rmin:rmax, cmin:cmax]
        img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
        img_masked = torch.as_tensor(img_masked).cuda()
        img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

        choose = torch.LongTensor(choose.astype(np.int32)).cuda()
        ycb_class_index = torch.LongTensor(
            [class_id - 2]
        ).cuda()  # ycb ids starts from 1
        pred_r, pred_t, pred_c, emb = estimator["pose_estimator"](
            img_masked, cloud, choose, ycb_class_index
        )
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)
        points = cloud.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)
        pred_wo_refine[obj] = my_pred.tolist()
        for ite in range(0, 10):
            T = (
                torch.from_numpy(my_t.astype(np.float32))
                .cuda()
                .view(1, 3)
                .repeat(num_points, 1)
                .contiguous()
                .view(1, num_points, 3)
            )
            my_mat = quaternion_matrix(my_r)
            R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t

            new_cloud = torch.bmm((cloud - T), R).contiguous()
            pred_r, pred_t = estimator["pose_refiner"](new_cloud, emb, ycb_class_index)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)

            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array(
                [my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]]
            )

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

            pred_wo_refine[obj] = my_pred.tolist()

    return pred_wo_refine


def get_mask(img, mask_estimator, xyz=None, vis=False):
    if xyz is not None:  # uois
        image_standardized = np.zeros_like(img).astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(3):
            image_standardized[..., i] = (img[..., i] / 255.0 - mean[i]) / std[i]
        batch = {
            "rgb": torch.from_numpy(image_standardized)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float(),  # .to(mask_estimator.device),
            "xyz": torch.from_numpy(xyz)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float(),  # .to(mask_estimator.device),
        }
        (
            fg_masks,
            center_offsets,
            initial_masks,
            seg_masks,
        ) = mask_estimator.run_on_batch(batch)
        segment = seg_masks.cpu().numpy()[0]
        if vis:
            from src.util.utilities import get_color_mask  # import from uois
            from src.util.utilities import subplotter, torch_to_numpy

            # visualization for uois
            # Get results in numpy
            seg_mask = seg_masks[0].cpu().numpy()
            fg_mask = fg_masks[0].cpu().numpy()
            center_offset = center_offsets[0].cpu().numpy().transpose(1, 2, 0)
            initial_mask = initial_masks[0].cpu().numpy()
            rgb_imgs = torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)
            total_subplots = 6
            fig_index = 1
            num_objs = np.unique(seg_mask).max() + 1
            rgb = rgb_imgs[0].astype(np.uint8)
            depth = xyz[..., 2]
            seg_mask_plot = get_color_mask(seg_mask, nc=num_objs)
            images = [rgb, depth, get_color_mask(fg_mask), seg_mask_plot]
            titles = [
                f"Image",
                "depth",
                "floor&table",
                f"Refined Masks. #objects: {np.unique(seg_mask).shape[0]-1}",
            ]
            fig = subplotter(images, titles, fig_num=1)
            plt.show()
            # plt.savefig('debug/uois.png')
        id2class = None  # uois is class agnostic
    else:  # maskrcnn
        segment = np.zeros(img.shape[:2])
        outputs = mask_estimator(
            img[:, :, ::-1]
        )  # NOTE: maskrcnn is trained with BGR images. img in param is RGB.
        masks = outputs["instances"].pred_masks.detach().cpu().numpy()
        classes = outputs["instances"].pred_classes.detach().cpu().numpy()
        del outputs
        id2class = {}
        for i, cls in enumerate(classes):
            segment[masks[i]] = i
            id2class[i] = cls
    return segment.astype(np.uint8), id2class


def main(time_step=0.01):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--vis_segment", action="store_true", help="visualize segmentation masks"
    )
    parser.add_argument(
        "-s", "--segment", action="store_true", help="use dnn for segmentation"
    )
    parser.add_argument(
        "-sn",
        "--segment_network",
        default="maskrcnn",
        help="choice of network for segmentation. maskrcnn | uois",
    )
    parser.add_argument(
        "-p", "--pose", action="store_true", help="use dnn for 6d pose estimation"
    )
    args = parser.parse_args()

    init_maskrcnn, init_uois, init_densefusion = False, False, False
    if args.segment:
        if args.segment_network == "maskrcnn":
            init_maskrcnn = True
        elif args.segment_network == "uois":
            init_uois = True
        else:
            raise KeyError(
                f"segmentation network has to be maskrcnn or uois but get {args.segment_network}"
            )
        print(f"Use {args.segment_network} for segmentation")
    if args.pose:
        init_densefusion = True
        from lib.transformations import (quaternion_from_matrix,
                                         quaternion_matrix)

        print(f"Use DenseFusion for 6D pose estimation")

    vis_handler = init_vision_utils(init_maskrcnn, init_uois, init_densefusion)

    real_world = connect(use_gui=True)
    p.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraYaw=65,
        cameraPitch=-30,
        cameraTargetPosition=[1.1, 0, 0.6],
    )
    add_data_path()

    ##################################################
    # generate room layout
    ##################################################

    floor = create_floor()

    table_len = 0.5
    table_h = 0.7
    table = create_box(table_len, table_len * 2, table_h, color=(0.75, 0.75, 0.75, 1.0))
    set_point(table, (0.75, 0, table_h / 2))

    mass = 1
    obj_num = np.random.randint(2) + 5

    objlist_ = os.listdir(YCB_BANK_DIR)
    objlist = [x for x in objlist_ if x[0] == "0"]
    objlist.sort()

    enable_gravity()

    box_shapes = np.random.choice(len(objlist), obj_num, replace=False)
    id2class = {}
    id2class[0] = 0  # floor
    id2class[1] = 1  # table
    for i in range(obj_num):
        angle = tuple(np.random.rand(3))
        # box_shape = np.random.randint(len(objlist))
        box_shape = box_shapes[i]
        box_vis = p.createVisualShape(
            p.GEOM_MESH, fileName=f"{YCB_BANK_DIR}/{objlist[box_shape]}/textured.obj"
        )
        box_col = p.createCollisionShape(
            p.GEOM_MESH, fileName=f"{YCB_BANK_DIR}/{objlist[box_shape]}/textured.obj"
        )
        box = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=box_col,
            baseVisualShapeIndex=box_vis,
            physicsClientId=real_world,
        )

        loc = (
            (np.random.rand(1)[0] - 0.5) * table_len / 2.0 + 0.75,
            (np.random.rand(1)[0] - 0.5) * table_len / 2.0,
            table_h + 0.2,
        )

        set_point(box, loc)
        set_euler(box, angle)

        for _ in range(200):
            p.stepSimulation()
        id2class[box] = (
            box_shape + 2
        )  # reserve 0 for floor and 1 for table. ycb object index starts at 2

    ###################################################

    gt_pose = []
    for i in range(obj_num):
        gt_pose.append(get_pose(i + 2))

    step = 0

    camera_pose = (0.15, 0.15, 1.15)
    target_pose = (0.75, 0, 0.7)

    (rgba, depth, segment), view_matrix, projection_matrix, _, _ = get_image(
        camera_pose, target_pose, segment=True
    )

    proj_matrix_real = np.asarray(projection_matrix).reshape(4, 4).transpose(1, 0)
    view_matrix_real = np.asarray(view_matrix).reshape(4, 4).transpose(1, 0)

    depth_scaled = (depth - 0.5) * 2  # scale from 0-1 to -1~1
    xmap = np.array([[i for i in range(640)] for j in range(480)]) / 640.0 * 2 - 1.0
    ymap = (
        np.array([[480 - j for i in range(640)] for j in range(480)]) / 480.0 * 2 - 1.0
    )
    normed_loc = np.concatenate(
        (
            xmap.reshape(1, -1),
            ymap.reshape(1, -1),
            depth_scaled.reshape(1, -1),
            np.ones((1, 640 * 480)),
        )
    )
    # loc in normalized space(-1~1)
    eye_loc = np.linalg.inv(proj_matrix_real).dot(normed_loc)
    eye_loc = (
        (eye_loc / eye_loc[-1])[:3, :].reshape(3, 480, 640).transpose(1, 2, 0)
    )  # H x W x 3
    if args.segment and args.segment_network == "maskrcnn":
        segment, id2class = get_mask(rgba[:, :, :3], vis_handler["mask"])
    elif args.segment and args.segment_network == "uois":
        uois_inputdep = eye_loc
        uois_inputdep[:, :, 2] *= -1
        uois_inputdep[uois_inputdep[:, :, 2] > 10.0] = 0  # clamp large values
        segment, id2class = get_mask(
            rgba[:, :, :3], vis_handler["mask"], xyz=uois_inputdep, vis=args.vis_segment
        )

    if args.vis_segment:
        plt.subplot(1, 2, 1)
        plt.imshow(rgba)
        plt.subplot(1, 2, 2)
        plt.imshow(segment)
        plt.show()

    debug = False  # check depth im
    if debug:
        cam_fx = proj_matrix_real[0, 0] * img_length / 2.0
        cam_fy = proj_matrix_real[1, 1] * img_width / 2.0
        cam_cx = (proj_matrix_real[2, 0] - 1.0) * img_length / -2.0
        cam_cy = (proj_matrix_real[2, 1] + 1.0) * img_width / 2.0
        cam_mat = np.matrix([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])

        if not os.path.exists("debug"):
            os.makedirs("debug")
        plt.imsave(f"debug/seg_{i}.png", segment * 40)
        plt.imsave(f"debug/rgba_{i}.png", rgba)
        plt.imsave(f"debug/dep_{i}.png", depth)
        print(f"save segmentation results to debug/")

    if not args.pose:
        return

    object_list = np.unique(segment)
    n_object = object_list.shape[0]
    print(f"{n_object} objects in the scene")

    if id2class is None:
        raise ValueError(
            "class label is needed for 6d pose estimation. (uois is class agnostic)"
        )

    pose6ds = get_pose6d(
        rgba,
        depth,
        eye_loc,
        segment,
        vis_handler,
        view_matrix_real,
        proj_matrix_real,
        id2class=id2class,
    )

    virtual_table = create_box(
        table_len, table_len * 2, table_h, color=(0.75, 0.75, 0.75, 0.6)
    )
    set_point(virtual_table, (0.75 + 0.75, 0, table_h / 2))

    cad_model = {}
    for obj_id in pose6ds.keys():
        box_vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=f"{YCB_BANK_DIR}/{objlist[id2class[obj_id]-2]}/textured.obj",
            specularColor=[0.5, 0, 0.5],
        )
        box_col = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=f"{YCB_BANK_DIR}/{objlist[id2class[obj_id]-2]}/textured.obj",
        )
        box = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=box_col,
            baseVisualShapeIndex=box_vis,
            physicsClientId=real_world,
        )
        mat_eye = quaternion_matrix(pose6ds[obj_id][:4])
        mat_eye[0:3, 3] = pose6ds[obj_id][-3:]
        mat_eye[2, :] *= -1  # The positive direction of Y and Z axis is
        # different for OpenGL coordinate and DenseFusion
        # coordinate
        mat_eye[1, :] *= -1

        mat_world = np.linalg.inv(view_matrix_real).dot(mat_eye)
        # import pdb;pdb.set_trace()
        tran_world = tuple((mat_world[0, 3], mat_world[1, 3], mat_world[2, 3]))
        mat_world[0:3, 3] = 0
        qua = quaternion_from_matrix(mat_world, True)
        quat_world = (qua[1], qua[2], qua[3], qua[0])

        set_point(box, tuple(np.asarray(tran_world) + np.asarray([0.75, 0, 0])))
        set_quat(box, quat_world)

    wait_for_user()
    disconnect()


if __name__ == "__main__":
    main()
