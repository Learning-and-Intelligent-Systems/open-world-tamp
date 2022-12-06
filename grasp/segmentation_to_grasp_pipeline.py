#!/usr/bin/env python

from __future__ import print_function

# NOTE(caelan): must come before other imports
import sys

sys.path.extend(
    [
        "pddlstream",
        "ss-pybullet",
        #'pddlstream/examples/pybullet/utils',
    ]
)
try:
    import pybullet as p
except ImportError:
    raise ImportError(
        "This example requires PyBullet (https://pypi.org/project/pybullet/)"
    )

import argparse
import cProfile
import os
import pstats
from collections import Counter, namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch
import torchvision.transforms as transforms
from examples.discrete_belief.run import MAX_COST, clip_cost, revisit_mdp_cost
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    Attach,
    Conf,
    Detach,
    Trajectory,
    apply_commands,
    get_base_limits,
    get_grasp_gen,
    get_ik_ir_gen,
    get_motion_gen,
    get_stable_gen,
)
from examples.pybullet.utils.pybullet_tools.pr2_problems import (
    create_box,
    create_floor,
    create_table,
)
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    ARM_NAMES,
    DRAKE_PR2_URDF,
    PR2_CAMERA_MATRIX,
    PR2_GROUPS,
    REST_LEFT_ARM,
    attach_viewcone,
    get_arm_joints,
    get_group_conf,
    get_group_joints,
    get_link_pose,
    get_pr2_field_of_view,
    get_viewcone,
    is_drake_pr2,
    pixel_from_ray,
    ray_from_pixel,
    rightarm_from_leftarm,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    AABB,
    BLACK,
    GREY,
    PI,
    RED,
    TAN,
    TEMP_DIR,
    WHITE,
    ClientSaver,
    HideOutput,
    LockRenderer,
    Pose,
    WorldSaver,
    aabb_from_points,
    add_body_name,
    add_data_path,
    add_line,
    apply_alpha,
    clone_world,
    connect,
    create_box,
    create_cylinder,
    create_mesh,
    create_obj,
    create_plane,
    disable_gravity,
    disconnect,
    draw_aabb,
    draw_base_limits,
    draw_mesh,
    draw_point,
    draw_pose,
    enable_gravity,
    get_aabb,
    get_aabb_center,
    get_bodies,
    get_camera,
    get_configuration,
    get_distance,
    get_joint_position,
    get_joint_positions,
    get_link_pose,
    get_pose,
    image_from_segmented,
    invert,
    is_center_stable,
    joints_from_names,
    link_from_name,
    load_model,
    load_pybullet,
    mesh_from_points,
    multiply,
    pairwise_collision,
    point_from_pose,
    pose_from_tform,
    quat_from_euler,
    remove_body,
    save_image,
    set_camera,
    set_camera_pose,
    set_camera_pose2,
    set_client,
    set_configuration,
    set_euler,
    set_joint_positions,
    set_point,
    set_pose,
    set_quat,
    stable_z,
    step_simulation,
    tform_point,
    unit_pose,
    wait_for_user,
    wait_if_gui,
)
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.search import ABSTRIPSLayer
from pddlstream.language.constants import And, Equal, Or, PDDLProblem, print_solution
from pddlstream.language.generator import (
    accelerate_list_gen_fn,
    fn_from_constant,
    from_fn,
    from_gen_fn,
    from_list_fn,
    from_test,
)
from pddlstream.language.stream import StreamInfo
from pddlstream.utils import get_file_path, read
from pybullet_tools.utils import get_image
from pybullet_tools.voxels import MAX_PIXEL_VALUE

from grasp.graspnet_interface import visualize_grasps
from vision_utils.test_vis_clean.constant import POSE_DIR, SEG_DIR, YCB_BANK_DIR
from vision_utils.test_vis_clean.primitives import (
    REG_RANGE,
    VIS_RANGE,
    Detect,
    Mark,
    Observe,
    Register,
    Scan,
    ScanRoom,
    get_cone_commands,
    get_fo_test,
    get_in_range_test,
    get_inverse_visibility_fixbase_fn,
    get_inverse_visibility_fn,
    get_isGraspable_test,
    get_isTarget_test,
    get_unblock_test,
    get_vis_base_gen,
    get_visclear_test,
    move_look_trajectory,
    plan_head_traj,
)
from vision_utils.test_vis_clean.problems import (
    USE_DRAKE_PR2,
    Voxelgrid,
    create_pr2,
    get_problem1,
)
from vision_utils.test_vis_clean.utils import ICC

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
MODEL_PATH = "./models"
obj_name = "003_cracker_box"
LTAMP_PR2 = os.path.join(MODEL_PATH, "pr2_description/pr2.urdf")
SUGAR_PATH = os.path.join(MODEL_PATH, "ycb/" + obj_name + "/textured.obj")

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

CAMERA_FRAME = "head_mount_kinect_rgb_link"
CAMERA_OPTICAL_FRAME = "head_mount_kinect_rgb_optical_frame"
WIDTH, HEIGHT = 640, 480
FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
# INTRINSICS = get_camera_matrix(WIDTH, HEIGHT, FX, FY)
INTRINSICS = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

LabeledPoint = namedtuple("LabledPoint", ["point", "color", "body"])


def init_vision_utils():
    """Detection & Segmentation - MaskRCNN"""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 23
    cfg.MODEL.WEIGHTS = "./vision_utils/pretrained_detectron2_model/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.7  # set the testing threshold for this model
    )
    predictor = DefaultPredictor(cfg)

    """ 6d Pose """
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
    toolbox["pose_estimator"] = estimator
    toolbox["pose_refiner"] = refiner
    toolbox["mask"] = predictor

    return toolbox


import sys

sys.path.append(POSE_DIR)
import copy

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_from_matrix, quaternion_matrix


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


def set_group_positions(pr2, group_positions):
    for group, positions in group_positions.items():
        joints = joints_from_names(pr2, PR2_GROUPS[group])
        assert len(joints) == len(positions)
        set_joint_positions(pr2, joints, positions)


def get_mask(img, mask_estimator):
    segment = np.zeros(img.shape[:2])
    outputs = mask_estimator(
        img[:, :, ::-1]
    )  # NOTE: maskrcnn is trained with BGR images. img in param is RGB.
    masks = outputs["instances"].pred_masks.detach().cpu().numpy()
    classes = outputs["instances"].pred_classes.detach().cpu().numpy()
    del outputs
    for i, cls in enumerate(classes):
        segment[masks[i]] = cls
    return segment.astype(np.int32)


seg_gt_list = {}  # TODO adhoc global variable


def main(time_step=0.01):

    parser = argparse.ArgumentParser()
    vis_handler = init_vision_utils()
    # parser.add_argument('-c', '--cfree', action='store_true',
    #                     help='When enabled, disables collision checking (for debugging).')
    # parser.add_argument('-p', '--problem', default='test_pour', choices=sorted(problem_fn_from_name),
    #                     help='The name of the problem to solve.')
    # parser.add_argument('-s', '--seed', default=None,
    #                     help='The random seed to use.')
    parser.add_argument("-v", "--viewer", action="store_true", help="")
    args = parser.parse_args()

    connect(use_gui=args.viewer)
    draw_pose(Pose(), length=1)
    with HideOutput():  # enable=None):
        add_data_path()
        # floor = create_floor()
        floor = create_plane(color=TAN)
        table = create_table(leg_color=GREY, cylinder=False)
        # table = load_pybullet(TABLE_URDF)
        obj = create_obj(SUGAR_PATH, color=WHITE)  # , **kwargs)
        pr2 = load_pybullet(LTAMP_PR2)

    set_pose(obj, ([0, 0, stable_z(obj, table)], p.getQuaternionFromEuler([0, 0, 1])))
    # dump_body(pr2)
    group_positions = {
        "base": [-1.0, 0, 0],
        "left_arm": REST_LEFT_ARM,
        "right_arm": rightarm_from_leftarm(REST_LEFT_ARM),
        "torso": [0.2],
        "head": [0, PI / 4],
    }
    set_group_positions(pr2, group_positions)
    # wait_if_gui()

    table_aabb = get_aabb(table)
    draw_aabb(table_aabb)
    x, y, _ = get_aabb_center(table_aabb)
    # target_point = [x, y, table_aabb[1][2]]

    camera_matrix = INTRINSICS
    camera_link = link_from_name(pr2, CAMERA_OPTICAL_FRAME)
    camera_pose = get_link_pose(pr2, camera_link)
    draw_pose(camera_pose, length=1)  # TODO: attach to camera_link

    distance = 2.5
    target_point = tform_point(camera_pose, np.array([0, 0, distance]))
    attach_viewcone(
        pr2,
        depth=distance,
        head_name=CAMERA_FRAME,
        camera_matrix=camera_matrix,
        color=RED,
    )
    # view_cone = get_viewcone(depth=distance, camera_matrix=camera_matrix, color=apply_alpha(RED, alpha=0.1))
    # set_pose(view_cone, camera_pose)

    # set_camera_pose2(camera_pose, distance)
    # camera_info = get_camera()
    # wait_if_gui()
    # return

    # TODO: OpenCV
    camera_pose = (0.15, 0.15, 1.15)
    camera_point = point_from_pose(camera_pose)
    _, vertical_fov = get_pr2_field_of_view(camera_matrix=camera_matrix)
    # view_matrix = p.computeViewMatrix(cameraEyePosition=camera_point, cameraTargetPosition=target_point,
    #                                  cameraUpVector=[0, 0, 1]) #, physicsClientId=CLIENT)
    # view_pose = pose_from_tform(np.reshape(view_matrix, [4, 4]))
    # projection_matrix = get_projection_matrix(width, height, vertical_fov, near, far)

    camera_pose = get_link_pose(pr2, camera_link)
    draw_pose(camera_pose, length=1)  # TODO: attach to camera_link

    distance = 2.5
    target_point = point_from_pose(get_pose(obj))
    attach_viewcone(
        pr2,
        depth=distance,
        head_name=CAMERA_FRAME,
        camera_matrix=camera_matrix,
        color=RED,
    )
    # view_cone = get_viewcone(depth=distance, camera_matrix=camera_matrix, color=apply_alpha(RED, alpha=0.1))
    # set_pose(view_cone, camera_pose)

    # set_camera_pose2(camera_pose, distance)
    # camera_info = get_camera()
    # wait_if_gui()
    # return

    # TODO: OpenCV
    cp1 = point_from_pose(camera_pose)
    camera_point = cp1
    labeled_points = []

    _, vertical_fov = get_pr2_field_of_view(camera_matrix=camera_matrix)

    camera_image = get_image(
        camera_pos=camera_point,
        target_pos=target_point,
        width=WIDTH,
        height=HEIGHT,
        vertical_fov=vertical_fov,
        tiny=False,
        segment=True,
        segment_links=False,
    )
    rgb_image, depth_image, seg_image = (
        camera_image.rgbPixels,
        camera_image.depthPixels,
        camera_image.segmentationMaskBuffer,
    )

    save_image(os.path.join(TEMP_DIR, str(0) + "_rgb.png"), rgb_image)  # [0, 255]
    save_image(os.path.join(TEMP_DIR, str(0) + "_depth.png"), depth_image)  # [0, 1]
    if seg_image is not None:
        segmented_image = image_from_segmented(seg_image, color_from_body=None)
        save_image(os.path.join(TEMP_DIR, "segmented.png"), segmented_image)  # [0, 1]

    step = 0
    segment = get_mask(rgb_image[:, :, :3], vis_handler["mask"])
    labeled_points = []

    step_size = 1
    for r in range(0, depth_image.shape[0], step_size):
        for c in range(0, depth_image.shape[1], step_size):
            if segment[r, c]:
                pixel = [c, r]  # NOTE: width, height
                # print(pixel, pixel_from_ray(camera_matrix, ray_from_pixel(camera_matrix, pixel)))
                ray = ray_from_pixel(camera_matrix, pixel)
                depth = depth_image[r, c]
                point_camera = depth * ray
                point_world = tform_point(multiply(camera_pose), point_camera)
                point_world = [point_world[1], point_world[2], point_world[0]]
                color = rgb_image[r, c, :] / MAX_PIXEL_VALUE
                labeled_points.append(LabeledPoint(point_world, color, obj))

    points = [point.point for point in labeled_points]
    mean_color = np.average(
        [point.color for point in labeled_points], axis=0
    )  # average | median
    centroid = np.average(points, axis=0)
    obj_aabb = aabb_from_points(points)
    origin_point = centroid
    origin_point[2] = obj_aabb[0][2]
    origin_pose = Pose(origin_point)  # PCA
    draw_pose(origin_pose)

    # plot the full 3d point cloud in mayavi
    pc = np.array([p.point for p in labeled_points])
    pc_color = np.array([p.color for p in labeled_points])[:, :3] * 255

    # rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
    # rgba[:, :3] = np.asarray(pc_color)
    # rgba[:, 3] = 255
    # src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    # src.add_attribute(rgba, 'colors')
    # src.data.point_data.set_active_scalars('colors')
    # g = mlab.pipeline.glyph(src)
    # g.glyph.scale_mode = "data_scaling_off"
    # g.glyph.glyph.scale_factor = 0.01
    # mlab.show()

    # Pass through the graspnet for inference
    visualize_grasps(pc, pc_color, obj_name)

    sys.exit(1)

    wait_for_user()
    disconnect()


if __name__ == "__main__":
    main()
