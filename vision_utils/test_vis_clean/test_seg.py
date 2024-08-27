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
                                  remove_body, save_image, set_client,
                                  set_euler, set_joint_positions, set_point,
                                  set_pose, set_quat, step_simulation,
                                  wait_for_user)
from src.evaluation import multilabel_metrics
from src.util.utilities import subplotter  # import from uois
from src.util.utilities import get_color_mask, torch_to_numpy

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
from lib.transformations import quaternion_from_matrix, quaternion_matrix


def init_vision_utils(
    init_maskrcnn=False,
    init_uois=False,
    usetodrrn=False,
    dsn_epsilon=0.05,
    dsn_sigma=0.02,
):
    toolbox = {}
    """Detection & Segmentation."""
    if init_maskrcnn:
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
        predictor = DefaultPredictor(cfg)
        toolbox["maskrcnn"] = predictor
    """ segmentation - UOIS """
    if init_uois:
        import src.segmentation as uois_segmentation

        from .constant import dsn_config, rrn_config, uois3d_config

        checkpoint_dir = os.path.join(UOIS_DIR, "checkpoints")
        dsn_filename = os.path.join(
            checkpoint_dir, "DepthSeedingNetwork_3D_TOD_checkpoint.pth"
        )
        rrn_filename = os.path.join(checkpoint_dir, "RRN_OID_checkpoint.pth")
        rrn_filename_tod = os.path.join(checkpoint_dir, "RRN_TOD_checkpoint.pth")
        uois3d_config["final_close_morphology"] = "TableTop_v5" in rrn_filename
        uois_predictor = uois_segmentation.UOISNet3D(
            uois3d_config, dsn_filename, dsn_config, rrn_filename, rrn_config
        )
        toolbox["uois"] = uois_predictor
        if dsn_epsilon != 0.05:  # if not default value
            dsn_config["epsilon"] = dsn_epsilon
            uois_predictor_epsilon = uois_segmentation.UOISNet3D(
                uois3d_config, dsn_filename, dsn_config, rrn_filename, rrn_config
            )
            dsn_config["epsilon"] = 0.05
            toolbox["uoiseps"] = uois_predictor_epsilon
        if dsn_sigma != 0.02:  # if not default value
            dsn_config["sigma"] = dsn_sigma
            uois_predictor_sigma = uois_segmentation.UOISNet3D(
                uois3d_config, dsn_filename, dsn_config, rrn_filename, rrn_config
            )
            dsn_config["sigma"] = 0.02
            toolbox["uoissig"] = uois_predictor_sigma
        if usetodrrn:  # if use checkpoint trained on synthetic dataset
            uois3d_config["final_close_morphology"] = "TableTop_v5" in rrn_filename_tod
            uois_predictor_todrrn = uois_segmentation.UOISNet3D(
                uois3d_config, dsn_filename, dsn_config, rrn_filename, rrn_config
            )
            toolbox["uoistod"] = uois_predictor_todrrn
    return toolbox


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
    else:
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
        "-viewer", action="store_true", help="enable the viewer while planning"
    )
    parser.add_argument(
        "-nr", "--num_rooms", default=1000, type=int, help="# table setups"
    )
    parser.add_argument(
        "-lst", "--least_obj_num", type=int, default=1, help="minimum number of objects"
    )
    parser.add_argument(
        "--random_angle",
        action="store_true",
        help="randomize the angle of objects before dropping",
    )
    parser.add_argument(
        "--freecam", action="store_true", help="camera can be located on top of table"
    )
    parser.add_argument(
        "--todrrn",
        action="store_true",
        help="use rrn checkpoint trained on tod(synthetic) dataset",
    )
    parser.add_argument(
        "-s",
        "--sigmaloc",
        type=float,
        default=0.2,
        help="sigma of dropping location of objects",
    )
    parser.add_argument(
        "-cs", "--sigmacam", type=float, default=1.2, help="sigma of camera location"
    )
    parser.add_argument(
        "-d",
        "--save_dir",
        type=str,
        default="debug/test_seg",
        help="dir to save results and log",
    )
    parser.add_argument(
        "-hc",
        "--cam_height",
        type=float,
        default=1.15,
        help="height of camera location",
    )
    parser.add_argument("--dsn_epsilon", type=float, default=0.05, help="dsn epsilon")
    parser.add_argument("--dsn_sigma", type=float, default=0.02, help="dsn sigma ")
    parser.add_argument(
        "-m", "--maskrcnn", action="store_false", help="dont evaluate maskrcnn"
    )
    parser.add_argument("-u", "--uois", action="store_false", help="dont evaluate uois")
    args = parser.parse_args()

    ##################################################
    # initialize networks
    ##################################################

    vis_handler = init_vision_utils(
        usetodrrn=args.todrrn,
        dsn_epsilon=args.dsn_epsilon,
        dsn_sigma=args.dsn_sigma,
        init_maskrcnn=args.maskrcnn,
        init_uois=args.uois,
    )

    seg_algos = []
    if args.maskrcnn:
        seg_algos.append("maskrcnn")
    if args.uois:
        seg_algos.append("uois")
    if args.todrrn:
        seg_algos.append("uoistod")
    if args.dsn_sigma != 0.02:
        seg_algos.append("uoissig")
    if args.dsn_epsilon != 0.05:
        seg_algos.append("uoiseps")

    real_world = connect(use_gui=True)  # for better rendering
    add_data_path()

    ##################################################
    # generate room layout
    ##################################################

    floor = create_floor()

    table_len1 = 1.0
    table_len2 = 1.0
    table_h = 0.7
    table = create_box(table_len1, table_len2, table_h, color=(0.75, 0.75, 0.75, 1.0))
    set_point(table, (0, 0, table_h / 2))
    """arglist."""
    print("==> Args:")
    for arg in vars(args):
        print(f"{arg}\t {getattr(args, arg)}")
    num_rooms = args.num_rooms
    least_obj_num = args.least_obj_num
    random_angle = args.random_angle
    objloc_sigma = args.sigmaloc
    cam_sigma = args.sigmacam
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save a copy of testing script
    os.system(
        f"cp vision_utils/test_vis_clean/{os.path.basename(__file__)} {save_dir}/"
    )
    """"""
    objlist_ = os.listdir(YCB_BANK_DIR)
    objlist = [x for x in objlist_ if x[0] == "0"]
    objlist.sort()
    enable_gravity()
    obj_loaded = []
    mass = 1

    id2class_gt = {}
    id2class_gt[0] = 0  # floor
    id2class_gt[1] = 1  # table

    from .constant import maskrcnn_class2name

    maskrcnn_name2class = {v: k for (k, v) in maskrcnn_class2name.items()}
    for obj_name in objlist:
        box_vis = p.createVisualShape(
            p.GEOM_MESH, fileName=f"{YCB_BANK_DIR}/{obj_name}/textured.obj"
        )
        box_col = p.createCollisionShape(
            p.GEOM_MESH, fileName=f"{YCB_BANK_DIR}/{obj_name}/textured.obj"
        )
        box = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=box_col,
            baseVisualShapeIndex=box_vis,
            physicsClientId=real_world,
        )
        set_point(box, (100, 100, 150))
        obj_loaded.append(box)
        id2class_gt[box] = maskrcnn_name2class[
            obj_name
        ]  # reserve 0 for floor and 1 for table. ycb object index starts at 2

    res_dict = {}

    record_all = {}
    best_ten = {}
    worst_ten = {}
    random_ten = {}
    best_fscore = {}
    worst_fscore = {}
    for seg_algo in seg_algos:
        record_all[seg_algo] = []
        best_ten[seg_algo] = []
        worst_ten[seg_algo] = []
        random_ten[seg_algo] = []
        best_fscore[seg_algo] = -200
        worst_fscore[seg_algo] = 200
    for room_i in range(num_rooms):
        obj_pybullet_list = []
        obj_num = np.random.randint(4) + least_obj_num

        box_shapes = np.random.choice(len(obj_loaded), obj_num, replace=False)
        for i in range(obj_num):
            if random_angle:
                angle = tuple(np.random.rand(3))
            else:
                angle = (0, 0, 0)
            box_shape = box_shapes[i]

            loc = (
                (np.random.rand(1)[0] - 0.5) * table_len1 * objloc_sigma,
                (np.random.rand(1)[0] - 0.5) * table_len2 * objloc_sigma,
                table_h + 0.2,
            )

            box = obj_loaded[box_shape]
            set_point(box, loc)
            set_euler(box, angle)

            for _ in range(200):
                p.stepSimulation()
            obj_pybullet_list.append(box)
        ####    ###############################################

        camera_posex = (np.random.rand(1)[0] - 0.5) * table_len1 * cam_sigma
        if args.freecam or np.abs(camera_posex) > table_len1 / 2.0:
            camera_posey = (np.random.rand(1)[0] - 0.5) * table_len2 * cam_sigma
        else:  # sample locations around the table(not above)
            camera_posey = (
                (1 + np.random.rand(1)[0] * (cam_sigma - 1)) * table_len2 * 0.5
            )
            camera_posey *= 1 if camera_posex > 0 else -1

        camera_pose = (camera_posex, camera_posey, args.cam_height)
        target_pose = (0, 0, table_h)

        (rgba, depth, segment_gt), view_matrix, projection_matrix, _, _ = get_image(
            camera_pose, target_pose, segment=True
        )
        proj_matrix_real = np.asarray(projection_matrix).reshape(4, 4).transpose(1, 0)
        view_matrix_real = np.asarray(view_matrix).reshape(4, 4).transpose(1, 0)
        depth_scaled = (depth - 0.5) * 2  # scale from 0-1 to -1~1
        im_h, im_w = rgba.shape[0:2]
        xmap = (
            np.array([[i for i in range(im_w)] for j in range(im_h)]) / im_w * 2 - 1.0
        )
        ymap = (
            np.array([[im_h - j for i in range(im_w)] for j in range(im_h)]) / im_h * 2
            - 1.0
        )
        normed_loc = np.concatenate(
            (
                xmap.reshape(1, -1),
                ymap.reshape(1, -1),
                depth_scaled.reshape(1, -1),
                np.ones((1, im_h * im_w)),
            )
        )
        # loc in normalized space(-1~1)
        eye_loc = np.linalg.inv(proj_matrix_real).dot(normed_loc)
        eye_loc = (
            (eye_loc / eye_loc[-1])[:3, :].reshape(3, 480, 640).transpose(1, 2, 0)
        )  # H x W x 3
        uois_inputdep = (
            eye_loc  # x,y,z. in meters. z: 0 for holes in depth images. farthest~10
        )
        uois_inputdep[:, :, 2] *= -1
        uois_inputdep[uois_inputdep[:, :, 2] > 6.0] = 0

        for segalgo in seg_algos:
            if "uois" in segalgo:
                xyzinp = uois_inputdep
            else:
                xyzinp = None
            segment, id2class = get_mask(
                rgba[:, :, :3], vis_handler[segalgo], xyz=xyzinp
            )
            metrics = multilabel_metrics(segment, segment_gt)
            record_all[segalgo].append(metrics)
            if (
                metrics["Objects F-measure"] != 0.0
                and metrics["Objects F-measure"] != 1.0
                and metrics["Objects F-measure"] > best_fscore[segalgo]
            ):
                if len(best_ten[segalgo]) < 10:
                    best_ten[segalgo].append(
                        [rgba, segment_gt, segment, metrics["Objects F-measure"]]
                    )
                else:
                    fscores = [it[3] for it in best_ten[segalgo]]
                    idxtoremove = np.argmin(fscores)
                    best_ten[segalgo][idxtoremove] = [
                        rgba,
                        segment_gt,
                        segment,
                        metrics["Objects F-measure"],
                    ]
                    best_fscore[segalgo] = min(fscores)
            if (
                metrics["Objects F-measure"] != 0.0
                and metrics["Objects F-measure"] != 1.0
                and metrics["Objects F-measure"] < worst_fscore[segalgo]
            ):
                if len(worst_ten[segalgo]) < 10:
                    worst_ten[segalgo].append(
                        [rgba, segment_gt, segment, metrics["Objects F-measure"]]
                    )
                else:
                    fscores = [it[3] for it in worst_ten[segalgo]]
                    idxtoremove = np.argmax(fscores)
                    worst_ten[segalgo][idxtoremove] = [
                        rgba,
                        segment_gt,
                        segment,
                        metrics["Objects F-measure"],
                    ]
                    worst_fscore[segalgo] = max(fscores)
            if len(random_ten[segalgo]) < 10:
                random_ten[segalgo].append([rgba, segment_gt, segment])

        for box in obj_loaded:
            set_point(box, (100, 100, 150))
            # remove_body(box)
        print(f"{room_i}/{num_rooms}")
    keynames = [
        "Objects F-measure",
        "Objects Precision",
        "Objects Recall",
        "Boundary F-measure",
        "Boundary Precision",
        "Boundary Recall",
        "obj_detected",
        "obj_detected_075",
        "obj_gt",
        "obj_detected_075_percentage",
    ]
    with open(f"{save_dir}/log", "w+") as logf:
        for arg in vars(args):
            print(f"{arg}\t {getattr(args, arg)}", file=logf)
        for segalgo in seg_algos:
            print("-" * 30)
            print(segalgo)
            print(segalgo, file=logf)
            allval = []
            print("".join(keynames))
            for keyname in keynames:
                val = np.mean([it[keyname] for it in record_all[segalgo]])
                print(f"{keyname} mean : {val}")
                print(f"{keyname} mean : {val}", file=logf)
                allval.append(val)
            print("".join([f"{val:.4f}\t" for val in allval]))
            print("".join([f"{val:.4f}\t" for val in allval]), file=logf)
            for i, (rgba, seggt, segpred, _) in enumerate(best_ten[segalgo]):
                save_image(f"{save_dir}/best_ten_{segalgo}_{i:02d}_rgb.png", rgba)
                save_image(
                    f"{save_dir}/best_ten_{segalgo}_{i:02d}_seggt.png",
                    get_color_mask(seggt),
                )
                seg_mask_plot = get_color_mask(segpred)
                save_image(
                    f"{save_dir}/best_ten_{segalgo}_{i:02d}_segpred.png", seg_mask_plot
                )
                mixed = (rgba[:, :, :3] * 0.7 + seg_mask_plot * 0.3).astype(np.uint8)
                save_image(f"{save_dir}/best_ten_{segalgo}_{i:02d}_pred.png", mixed)
            for i, (rgba, seggt, segpred, _) in enumerate(worst_ten[segalgo]):
                save_image(f"{save_dir}/worst_ten_{segalgo}_{i:02d}_rgb.png", rgba)
                save_image(
                    f"{save_dir}/worst_ten_{segalgo}_{i:02d}_seggt.png",
                    get_color_mask(seggt),
                )
                seg_mask_plot = get_color_mask(segpred)
                save_image(
                    f"{save_dir}/worst_ten_{segalgo}_{i:02d}_segpred.png", seg_mask_plot
                )
                mixed = (rgba[:, :, :3] * 0.7 + seg_mask_plot * 0.3).astype(np.uint8)
                save_image(f"{save_dir}/worst_ten_{segalgo}_{i:02d}_pred.png", mixed)
            for i, (rgba, seggt, segpred) in enumerate(random_ten[segalgo]):
                save_image(f"{save_dir}/random_ten_{segalgo}_{i:02d}_rgb.png", rgba)
                save_image(
                    f"{save_dir}/random_ten_{segalgo}_{i:02d}_seggt.png",
                    get_color_mask(seggt),
                )
                seg_mask_plot = get_color_mask(segpred)
                save_image(
                    f"{save_dir}/random_ten_{segalgo}_{i:02d}_segpred.png",
                    seg_mask_plot,
                )
                mixed = (rgba[:, :, :3] * 0.7 + seg_mask_plot * 0.3).astype(np.uint8)
                save_image(f"{save_dir}/random_ten_{segalgo}_{i:02d}_pred.png", mixed)
            print(f"{args.save_dir} finished.")
            print("-" * 30)
            print("-" * 30, file=logf)
    disconnect()


if __name__ == "__main__":
    main()
