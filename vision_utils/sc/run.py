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
import os
import pstats
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.extend(
    [
        "pddlstream",
        #'pybullet-planning',
        #'pddlstream/examples/pybullet/utils',
    ]
)
from examples.pybullet.utils.pybullet_tools.pr2_problems import create_floor
from examples.pybullet.utils.pybullet_tools.utils import (
    CLIENT,
    PI,
    CameraImage,
    add_data_path,
    connect,
    create_mesh,
    get_projection_matrix,
    invert_quat,
    load_pybullet,
    obj_file_from_mesh,
    quat_from_matrix,
    read_pcd_file,
    remove_body,
    set_point,
    set_quat,
    wait_for_user,
)

# """""""""""""""""""""""""""""""""""""
# EDITED FUNCTIONS
# """""""""""""""""""""""""""""""""""""


def get_image(
    camera_pos,
    target_pos,
    width=640,
    height=480,
    vertical_fov=PI / 3,
    near=0.01,
    far=1000.0,
    segment=False,
    segment_links=False,
    client=CLIENT,
):
    diff = np.asarray(list(target_pos)) - np.asarray(list(camera_pos))
    diff_len = (diff ** 2).sum() ** 0.5

    right = np.cross(diff, np.asarray([0, 0, 1]))
    up_vector = np.cross(right, diff)
    if (up_vector ** 2).sum() == 0:  # look perpendicular to the ground
        up_vector = np.asarray([1, 0, 0])

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vector,
        physicsClientId=client,
    )
    projection_matrix = get_projection_matrix(width, height, vertical_fov, near, far)

    if segment:
        if segment_links:
            flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        else:
            flags = 0
    else:
        flags = p.ER_NO_SEGMENTATION_MASK
    image = CameraImage(
        *p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            shadow=False,
            flags=flags,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client,
        )[2:]
    )
    depth = image.depthPixels
    segmented = image.segmentationMaskBuffer
    return (
        CameraImage(image.rgbPixels, depth, segmented),
        view_matrix,
        projection_matrix,
        width,
        height,
    )


class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()

        sys.path.append(ATLAS_PATH)
        from model.atlasnet import Atlasnet
        from model.model_blocks import (
            PointNet,
        )  # NOTE: if used together with DenseFusion, be careful about the class name

        self.encoder = PointNet(nlatent=opt.bottleneck_size)
        self.decoder = Atlasnet(opt)
        self.to(opt.device)
        self.eval()

    def forward(self, x, train=True):
        return self.decoder(self.encoder(x), train=train)

    def generate_mesh(self, x):
        atlas_list = self.decoder.generate_mesh(self.encoder(x))
        return atlas_list  # a list of atlas, each can be converted to a mesh


# """""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""


def init_vision_utils(base_path=None, ckpt_path=None):
    from collections import namedtuple

    atlas_opt = namedtuple(
        "atlasnet_opt",
        [
            "demo",
            "SVR",
            "reload_model_path",
            "nb_primitives",
            "template_type",
            "dim_template",
            "device",
            "bottleneck_size",
            "number_points",
            "number_points_eval",
            "remove_all_batchNorms",
            "hidden_neurons",
            "num_layers",
            "activation",
        ],
    )
    opt = atlas_opt(
        True,
        True,
        ckpt_path,
        1,
        "SPHERE",
        3,
        DEVICE,
        1024,
        2500,
        2500,
        False,
        512,
        2,
        "relu",
    )
    network = EncoderDecoder(opt)
    sdict = torch.load(opt.reload_model_path, map_location=DEVICE)

    from collections import OrderedDict

    new_dict = OrderedDict()
    for k, v in sdict.items():
        name = k[7:]
        new_dict[name] = v
    network.load_state_dict(new_dict)
    return network


def main(room_k=-1, time_step=0.01):
    real_world = connect(use_gui=True)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8124,
        cameraYaw=-40,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0.6],
    )  # just for visualization

    add_data_path()
    floor = create_floor()

    objlist = os.listdir(TEST_MODELS_PATH)
    objlist = [x for x in objlist if x[-3:] == "obj"]
    sc_network = init_vision_utils(ATLAS_PATH, CKPT_PATH)

    for box_shape in range(len(objlist)):
        loc = (0, 0, 0.65)
        color = np.random.rand(3)
        box_name = f"{TEST_MODELS_PATH}/{objlist[box_shape]}"
        print(box_name)
        box = load_pybullet(f"{TEST_MODELS_PATH}/{objlist[box_shape]}")
        set_point(box, loc)
        c_pose = np.asarray(
            [
                np.random.randn(1)[0],
                np.random.randn(1)[0],
                0.7 + 0.2 * np.random.rand(1)[0],
            ]
        )  # camera pose
        target_pose = (0, 0, 0.6)
        (rgba, depth, segment), view_matrix, projection_matrix, im_w, im_h = get_image(
            c_pose, target_pose, segment=True
        )
        visualize = False
        if visualize:
            plt.imshow(rgba)
            plt.show()
        with torch.no_grad():
            im = depth * 2 - 1.0
            mask = segment
            xmap = (
                np.array([[i for i in range(im_w)] for j in range(im_h)])
                / float(im_w)
                * 2
                - 1
            )
            ymap = (
                np.array([[im_h - j for i in range(im_w)] for j in range(im_h)])
                / float(im_h)
                * 2
                - 1
            )
            pc_incomplete = np.concatenate(
                (
                    xmap.reshape(1, -1),
                    ymap.reshape(1, -1),
                    im.reshape(1, -1),
                    np.ones((1, im_h * im_w)),
                )
            )
            pm = np.asarray(projection_matrix).reshape(4, 4).transpose(1, 0)

            eye_loc = np.linalg.inv(pm).dot(pc_incomplete)
            eye_loc = (
                (eye_loc / eye_loc[-1])[:3, :]
                .reshape(3, im_h, im_w)
                .transpose(1, 2, 0)
                .reshape(-1, 3)
            )  # partial point cloud in eye coordinate
            choose = np.where(mask.reshape(-1) == mask.max())[0]
            pc_incomplete = eye_loc[choose, :]
            mean = pc_incomplete.mean(0)
            pc_incomplete -= mean
            scale_fac = (pc_incomplete ** 2).sum(1).max() ** 0.5
            pc_incomplete = (pc_incomplete / scale_fac).transpose(1, 0)
            data = torch.Tensor(pc_incomplete).unsqueeze(0)
            network_input = data.to(DEVICE)
            mesh_list = sc_network.generate_mesh(network_input)
            shift = mean

            recon_list = []
            for submesh in mesh_list:
                submesh[0] *= scale_fac
                obj_recon = create_mesh(submesh)
                recon_list.append(obj_recon)

        vm = np.asarray(view_matrix).reshape(4, 4).transpose(1, 0)
        pm = np.asarray(projection_matrix).reshape(4, 4).transpose(1, 0)
        shift_eye = np.concatenate((shift, np.ones(1)))
        shift_eye = np.linalg.inv(vm).dot(shift_eye)
        shift_for_visualization = np.array(
            [0, 0, 0.3]
        )  # set to zero to see in the original location
        shift_eye = tuple((shift_eye / shift_eye[-1])[:3] + shift_for_visualization)
        for recon in recon_list:
            set_point(recon, shift_eye)
            set_quat(recon, invert_quat(quat_from_matrix(vm[:3, :3])))
        wait_for_user()
        for recon in recon_list:
            remove_body(recon)
        remove_body(box)
    wait_for_user()


DEVICE = torch.device("cpu")  # cpu | cuda | cuda:0
SC_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
ATLAS_PATH = os.path.join(SC_DIRECTORY, "../AtlasNet/")
CKPT_PATH = os.path.join(SC_DIRECTORY, "pc2pc.pth")
TEST_MODELS_PATH = os.path.join(SC_DIRECTORY, "bowls/")

if __name__ == "__main__":
    main(0)
