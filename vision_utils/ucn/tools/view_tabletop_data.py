# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import glob
import math
import os
import os.path
import random

import _init_paths
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import TableTopObject
from transforms3d.quaternions import mat2quat, quat2mat
from utils import mask as util_

if __name__ == "__main__":
    tabletop = TableTopObject("train")
    num = tabletop._size
    index = np.random.permutation(num)

    for idx in index:
        # Get scene directory, crop dose not use background
        scene_idx = idx // tabletop.NUM_VIEWS_PER_SCENE
        scene_dir = tabletop.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % tabletop.NUM_VIEWS_PER_SCENE

        # Label
        foreground_labels_filename = os.path.join(
            scene_dir, "segmentation_%05d.png" % view_num
        )
        # label = util_.imread_indexed(foreground_labels_filename)
        label = cv2.imread(foreground_labels_filename)

        # BGR image
        filename = os.path.join(scene_dir, "rgb_%05d.jpeg" % view_num)
        im = cv2.imread(filename)

        # Depth image
        depth_img_filename = os.path.join(scene_dir, "depth_%05d.png" % view_num)
        im_depth = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH)

        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.axis("off")

        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(im_depth)
        plt.axis("off")

        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(label[:, :, (2, 1, 0)])
        plt.axis("off")
        plt.show()
