from __future__ import print_function

import numpy as np
import open3d as o3d
from utils import utils

from grasp.graspnet import grasp_estimator
# from grasp.graspnet.utils.visualization_utils import *
from grasp.graspnet.utils import utils


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def backproject(
    depth_cv, intrinsic_matrix, return_finite_depth=True, return_selection=False
):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X


def score_grasps(estimator, pc, grasps):
    grasp_eulers, grasp_translations = utils.convert_qt_to_rt(grasps)
    grasp_pcs = utils.control_points_from_rot_and_trans(
        grasp_eulers, grasp_translations, estimator.device
    )
    results = estimator.grasp_evaluator.evaluate_grasps(pc, grasp_pcs)


# This function just generates grasps which can be fed into the score_grasps function
def generate_demo_grasps(estimator, pc):
    # Score grasps
    pc_list, pc_mean = estimator.prepare_pc(pc)
    grasps_list, confidence_list, z_list = estimator.generate_grasps(pc_list)
    for pcs, grasps in zip(pc_list, grasps_list):
        return pcs, grasps


kwargs = {
    "npy_folder": "grasp/graspnet/demo/data/",
    "grasp_sampler_folder": "grasp/graspnet/checkpoints/gan_pretrained/",
    "continue_train": False,
    "is_train": False,
    "grasp_evaluator_folder": "grasp/graspnet/checkpoints/evaluator_pretrained/",
    "target_pc_size": 1024,
    "num_grasp_samples": 200,
    "refine_steps": 25,
    "refinement_method": "sampling",
    "threshold": 0.8,
    "batch_size": 30,
    "generate_dense_grasps": False,
    "choose_fn": "better_than_threshold",
}
args = Struct(**kwargs)
grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
grasp_sampler_args.is_train = False
grasp_evaluator_args = utils.read_checkpoint_args(args.grasp_evaluator_folder)
grasp_evaluator_args.continue_train = True
estimator = grasp_estimator.GraspEstimator(
    grasp_sampler_args, grasp_evaluator_args, args
)


def generate_grasps(pc, pc_colors):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

    return estimator.generate_and_refine_grasps(pc[ind, :])


def visualize_grasps(pc, pc_colors, save_name):
    mayavi_data = {
        "pc": pc[ind, :],
        "pc_colors": pc_colors[ind, :],
        "grasps": generated_grasps,
        "grasp_scores": generated_scores,
    }
    write_to_minio(mayavi_data, "%s" % (save_name))
    generated_grasps, generated_scores = generate_grasps(pc, pc_colors)
    return estimator
