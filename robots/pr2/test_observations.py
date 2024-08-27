# TODO: the PR2 only supports python2 (due to ROS)

from __future__ import print_function

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# NOTE(caelan): must come before other imports
sys.path.extend(
    [
        "pybullet-planning",
    ]
)
from environments import \
    create_default_env  # Must be before importing TEMP_DIR
from open_world.estimation.belief import Belief
from open_world.estimation.dnn import init_sc, init_seg
from open_world.estimation.observation import image_from_labeled
from open_world.estimation.tables import estimate_surfaces
from open_world.simulation.entities import TABLE, get_label_counts
from pybullet_tools.utils import (SEPARATOR, CameraImage, connect, disconnect,
                                  elapsed_time, read_pickle, remove_body,
                                  save_image, user_input)
from run_estimator import cloud_from_depth

SURFACE_COLORS = {
    TABLE: [0.855, 0.733, 0.612],  # 0.596
    "red": [0.737, 0.082, 0.227],  # 0.044
    "yellow": [0.953, 0.89, 0.169],  # 0.043
    "green": [0.631, 0.902, 0.463],
    "blue": [0.31, 0.431, 0.804],
}

directory = "temp_meshes"


class Panda_Robot(object):
    def __init__(self):
        pass


# adhoc_pose = ((-0.321481317281723, -0.01055524218827486, 1.6538445949554443), (0.6829894781112671, -0.6809220314025879, 0.18662633001804352, -0.18719299137592316)) # default height of camera
plane_threshold = 1e-2

import torch

torch.manual_seed(0)
np.random.seed(0)


def main():
    parser = argparse.ArgumentParser()  # TODO: use create_parser
    parser.add_argument(
        "-p", "--path", default="data/run_21-04-16_17-30-18.pickle3", help=""
    )
    parser.add_argument("-m", "--model", default="all", help="")
    parser.add_argument(
        "-rgbd",
        "--maskrcnn_rgbd",
        action="store_true",
        help="set to True to use RGBD for maskrcnn",
    )
    parser.add_argument("-sc", "--shape_completion", action="store_true", help="")
    parser.add_argument(
        "-det",
        "--fasterrcnn_detection",
        action="store_true",
        help="use FasterRCNN to label the instances segmented by UOIS",
    )
    parser.add_argument(
        "-fd",
        "--fill_depth",
        action="store_true",
        help="apply maximum filter on depth image",
    )
    args = parser.parse_args()

    seg_network = init_seg(
        branch=args.model,
        maskrcnn_rgbd=args.maskrcnn_rgbd,
        post_classifier=args.fasterrcnn_detection,
    )
    if args.shape_completion:
        sc_network = init_sc(branch="msn")
    else:
        sc_network = None

    start_time = time.time()
    print("Loading", args.path)
    observations = read_pickle(args.path)["observations"]
    print(
        "Loaded {} observations in {:.3f} seconds".format(
            len(observations), elapsed_time(start_time)
        )
    )

    for i, observation in enumerate(observations):
        print(SEPARATOR)
        # import pdb;pdb.set_trace()
        # print(i, observation['frame'], observation.keys())

        rgb_image = observation["rgb"]
        print("RGB:", rgb_image.shape, np.min(rgb_image), np.max(rgb_image))
        save_image(os.path.join(directory, "real_rgb_{:02d}.png".format(i)), rgb_image)

        depth_image = observation["depth"]
        raw_depth = depth_image.copy()
        raw_depth[np.isnan(raw_depth)] = 0.0

        filtered_depth_image = depth_image[~np.isnan(depth_image)]  # TODO: clip
        print(
            "Depth:",
            depth_image.shape,
            np.min(filtered_depth_image),
            np.max(filtered_depth_image),
        )
        # depth_image /= np.max(filtered_depth_image)
        depth_image = (depth_image - np.min(filtered_depth_image)) / (
            np.max(filtered_depth_image) - np.min(filtered_depth_image)
        )
        save_image(
            os.path.join(directory, "real_depth_{:02d}.png".format(i)), depth_image
        )

        if args.fill_depth:
            from scipy.ndimage import maximum_filter

            for _ in range(2):
                # 1. maximum filter
                raw_depth_fil = maximum_filter(raw_depth, size=5)
                # # 2. median filter
                # raw_depth_fil = median_filter(raw_depth, size=5)

                # # 3. other
                # def fil_func(x):
                #     med = np.median(x[x!=0])
                #     return 0 if np.isnan(med) else med
                # raw_depth_fil = generic_filter(raw_depth, fil_func, size=5)
                raw_depth += (raw_depth == 0) * raw_depth_fil

        cloud = cloud_from_depth(observation["camera_matrix"], raw_depth)

        predicted_seg = seg_network.get_seg(
            rgb_image,
            point_cloud=cloud,
            depth_image=raw_depth,
            debug=True,
            relabel_fg=False,
            save_fig=False,
        )
        print("Predictions:", get_label_counts(predicted_seg))
        segmented_image = image_from_labeled(predicted_seg)
        save_image(
            os.path.join(
                directory,
                "{}_real_segmented_{:02d}.png".format(
                    os.path.basename(args.path).split(".")[0], i
                ),
            ),
            segmented_image,
        )  # [0, 255]
        plt.savefig(
            os.path.join(
                directory,
                "{}_segmented_{:02d}.png".format(
                    os.path.basename(args.path).split(".")[0], i
                ),
            )
        )
        plt.close()

        if args.shape_completion:
            camera_image = CameraImage(
                rgb_image,
                raw_depth,
                predicted_seg,
                observation["camera_pose"],
                observation["camera_matrix"],
            )
            connect()
            robot, table, _ = create_default_env()

            # pipeline on pr2
            remove_body(table)
            belief = Belief(robot, surface_beliefs=[])

            surfaces = estimate_surfaces(
                belief, camera_image, plane_threshold=plane_threshold
            )

            objects = belief.estimate_objects(
                camera_image,
                use_seg=True,
                surface=surfaces[0].surface,
                sc_network=sc_network,
                save_relabeled_img=True,
                concave=True,
                surfaces_movable=True,
            )

            # # vanilla
            # test_completion(sc_network, camera_image, [], concave=True,
            #                 use_instance_label=True, use_open3d=False, use_trimesh=True, noise_only=False)

            user_input("Continue?")
            disconnect()
        # break


if __name__ == "__main__":
    main()
