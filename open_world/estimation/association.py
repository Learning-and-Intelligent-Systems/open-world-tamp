import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_erosion
from open_world.estimation.geometry import estimate_surface_mesh
from open_world.estimation.observation import iterate_point_cloud, save_image_from_cluster, \
    aggregate_color, iterate_image, custom_iterate_point_cloud
from open_world.estimation.clustering import relabel_clusters
from pybullet_tools.utils import pixel_from_ray, tform_point, invert, multiply, \
        tform_points
from open_world.estimation.geometry import cloud_from_depth,\
    tform_point, invert
from open_world.simulation.entities import ENVIRONMENT

DIST_THRES = 3e-2
OVERLAP_THRES = 500  # 0 # TODO mem
IOU_THRES = 0.7

def get_predflow(camera_image_prev, posechange):
    rgb, dep, seg, _, _ = camera_image_prev
    before_poses, after_poses = posechange
    # before2after = tform_from_pose(after_poses[0]).dot(np.linalg.inv(tform_from_pose(before_poses[aa[0]])))
    import pdb

    pdb.set_trace()
    return rgb


def get_overlap_obj(immask, camera_image, objects, tform_poses):
    _, dep, segmask, camera_pose, camera_matrix = camera_image
    overlap_objs = []
    points_cameraframe_immask = cloud_from_depth(
        camera_matrix, dep, top_left_origin=True
    )  # .reshape(*dep.shape,3)[immask==1].reshape(-1,3)
    points_worldframe_immask = (
        np.asarray(tform_points(camera_pose, points_cameraframe_immask.reshape(-1, 3)))
        .reshape(*dep.shape, 3)[immask == 1]
        .reshape(-1, 3)
    )
    all_projmask = []
    for estimated_object in objects:
        tform_as_pred = tform_poses[estimated_object]
        mtform = multiply(invert(multiply(camera_pose)), tform_as_pred)
        pt_worldframe = [
            tform_point(mtform, lp.point) for lp in estimated_object.labeled_points
        ]
        pixels_cluster = [
            pixel_from_ray(camera_matrix, lp).astype(np.int32) for lp in pt_worldframe
        ]  # TODO use pixel_from_point
        # pixels_cluster = [pixel_from_ray(camera_matrix, tform_point(mtform, lp.point)).astype(np.int32) for lp in estimated_object.labeled_points]  # TODO use pixel_from_point
        im = np.zeros_like(dep)
        im_x, im_y = np.transpose(pixels_cluster)
        im_x[im_x >= dep.shape[1]] = 0
        im_x[im_x < 0] = 0
        im_y[im_y >= dep.shape[0]] = 0
        im_y[im_y < 0] = 0
        im[im_y, im_x] = 1
        im = binary_erosion(
            binary_closing(im, structure=np.ones((3, 3)), iterations=3),
            structure=np.ones((3, 3)),
        ).astype(np.int)
        # plt.imshow(im);plt.show();plt.close()
        # import pdb;pdb.set_trace()
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(np.vstack(pt_worldframe))
        distances, _ = neigh.kneighbors(points_worldframe_immask, return_distance=True)
        all_projmask.append(im)
        if (
            distances.ravel().min() < DIST_THRES
            and np.logical_and(immask, im).sum() > OVERLAP_THRES
            and np.logical_and(immask, im).sum() / im.sum() > IOU_THRES
        ):
            overlap_objs.append(estimated_object)
        else:
            print("min dist", distances.ravel().min())
            print("overlap sum ", np.logical_and(immask, im).sum())
            print("iou: ", np.logical_and(immask, im).sum() / im.sum())
    return all_projmask, overlap_objs


def get_overlap_objcloud(immask, camera_image, world_frame=True):
    _, dep, segmask, camera_pose, camera_matrix = camera_image
    cloud = cloud_from_depth(camera_matrix, dep, top_left_origin=True)
    if world_frame:
        cloud = tform_points(camera_pose, cloud.reshape(-1, 3))
        cloud = np.asarray(cloud).reshape(*dep.shape, 3)
    overlap_clouds, segname = [], []
    
    for seg_i in np.unique(segmask[..., 1]):
        env_overlap = np.vectorize(lambda x: x in ENVIRONMENT)(
            segmask[..., 0][segmask[..., 1] == seg_i]
        )
        mask_overlap = np.logical_and(immask, segmask[..., 1] == seg_i)

        if (
            mask_overlap.sum() > OVERLAP_THRES
            and mask_overlap.sum() / (segmask[..., 1] == seg_i).sum() > IOU_THRES
        ):
            overlap_clouds.append(cloud[segmask[..., 1] == seg_i].reshape(-1, 3))
            segname.append(seg_i)

    return cloud, overlap_clouds, segname


def compute_tform(cloud1, cloud2, use_open3d=False):
    # TODO asso use texture
    """
    cloud1: N x 3
    cloud2: M x 3
    """
    minpoint = min(cloud1.shape[0], cloud2.shape[0])
    cloud1 = cloud1 if cloud1.shape[0] == minpoint else sample_pc(cloud1, minpoint)
    cloud2 = cloud2 if cloud2.shape[0] == minpoint else sample_pc(cloud2, minpoint)
    if cloud1.shape != cloud2.shape:
        import pdb

        pdb.set_trace()
    tform, dist = icp(cloud1, cloud2)
    return tform, dist  # .mean()


# from filter_seg
def np_to_o3d(array):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(array))
    return pcd


# http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
# https://github.com/isl-org/Open3D/blob/master/examples/python/pipelines/global_registration.ipynb
# see pc_icp/wrong/global_reg.py for some exps
def preprocess_point_cloud(pcd, voxel_size, color=None):
    pcd_down = pcd
    sampled_idx = [i for i in range(len(np.asarray(pcd_down.points)))]

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    color_down = np.vstack([color[np.asarray(i)].mean(0) for i in sampled_idx])
    pcd_fpfh.data = np.concatenate((pcd_fpfh.data, color_down.T))  # pcd_fpfh.data=3xN
    # https://github.com/isl-org/Open3D/blob/master/cpp/open3d/pipelines/registration/Feature.cpp
    return pcd, pcd_down, pcd_fpfh


def global_reg(
    src, tgt, init_pose=None, voxel_size=5e-3, pt_prev_color=None, pt_cur_color=None
):
    source, source_down, source_fpfh = preprocess_point_cloud(
        np_to_o3d(src), voxel_size, color=pt_prev_color
    )
    target, target_down, target_fpfh = preprocess_point_cloud(
        np_to_o3d(tgt), voxel_size, color=pt_cur_color
    )
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, voxel_size, result.transformation
    )
    return (
        result.transformation,
        evaluation.fitness,
        evaluation.inlier_rmse ** 0.5,
    )  # fitness


# import cv2
def get_scene_flow(camera_image_obs, camera_image_pred, flow_network):
    # TODO incorporate observed/unobserved space
    # TODO view point

    # https://github.com/xingyul/flownet3d
    # https://github.com/princeton-vl/RAFT-3D
    rgb_1, depth_1, _, _, _ = camera_image_pred
    rgb_2, depth_2, _, _, _ = camera_image_obs
    rgb_1 = rgb_1[..., :3]
    rgb_2 = rgb_2[..., :3]
    preprocrgb = lambda x: torch.from_numpy(x).unsqueeze(0)
    flow_im = flow_network.forward(preprocrgb(rgb_1), preprocrgb(rgb_2))  # H W 2

    # visualization
    h, w = flow_im.shape[:2]
    flow_new = flow_im.copy()  # .astype(np.int64)
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]
    threshold = 2.0
    skip_amount = 50
    image = rgb_1.copy()
    flow_start = np.stack(
        np.meshgrid(range(flow_im.shape[1]), range(flow_im.shape[0])), 2
    )
    flow_end = (flow_im[flow_start[:, :, 1], flow_start[:, :, 0]] + flow_start).astype(
        np.int32
    )
    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0
    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for j in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][j], nz[1][j]
        cv2.arrowedLine(
            image,
            pt1=tuple(flow_start[y, x]),
            pt2=tuple(flow_end[y, x]),
            color=(0, 225, 0),
            thickness=1,
            tipLength=0.05,
        )
    import matplotlib.pyplot as plt

    plt.imshow(image * 0.7 / 255.0 + rgb_2.copy() / 255.0 * 0.3)
    plt.show()
    plt.close()
    import pdb

    pdb.set_trace()
    return flow_im


# def get_scene_flow(camera_image_obs, camera_image_pred, flow_network):
#     # TODO incorporate observed/unobserved space
#     # TODO view point

#     # https://github.com/xingyul/flownet3d
#     # https://github.com/princeton-vl/RAFT-3D
#     rgb_1, depth_1, _, _, _ = camera_image_obs
#     rgb_2, depth_2, _, _, _ = camera_image_pred
#     normdep = lambda x: torch.from_numpy((x-x.min())/(x.max()-x.min()) * 255).unsqueeze(0).unsqueeze(-1).repeat(1,1,1,3)
#     flow_im = flow_network.forward(normdep(depth_1), normdep(depth_2)) # H W 2

#     # # visualization
#     # h, w = flow_im.shape[:2]
#     # flow_new = flow_im.copy()#.astype(np.int64)
#     # flow_new[:,:,0] += np.arange(w)
#     # flow_new[:,:,1] += np.arange(h)[:,np.newaxis]
#     # threshold=2.
#     # skip_amount=50
#     # image = normdep(depth_1)[0].numpy().copy()
#     # flow_start = np.stack(np.meshgrid(range(flow_im.shape[1]), range(flow_im.shape[0])), 2)
#     # flow_end = (flow_im[flow_start[:,:,1],flow_start[:,:,0]] + flow_start).astype(np.int32)
#     # # Threshold values
#     # norm = np.linalg.norm(flow_end - flow_start, axis=2)
#     # norm[norm < threshold] = 0
#     # # Draw all the nonzero values
#     # nz = np.nonzero(norm)
#     # for j in range(0, len(nz[0]), skip_amount):
#     #     y, x = nz[0][j], nz[1][j]
#     #     cv2.arrowedLine(image,
#     #                     pt1=tuple(flow_start[y,x]),
#     #                     pt2=tuple(flow_end[y,x]),
#     #                     color=(0, 225, 0),
#     #                     thickness=1,
#     #                     tipLength=.05)
#     # import matplotlib.pyplot as plt
#     # plt.imshow(image*0.7/255.+normdep(depth_2)[0].numpy().copy()/255.*0.3);plt.show();plt.close()
#     # import pdb;pdb.set_trace()
#     return flow_im


def asso_mode_1(flow):
    if (flow ** 2).mean() < 1:
        return True
    else:
        return False


def asso_mode_2(camera_image_obs, camera_image_sim):
    return False


def asso_mode_3(camera_image_obs, camera_image_sim):
    return False


def asso_mode_4(camera_image_obs, camera_image_sim):
    return False


from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


# TODO asso. use chamfer dist
def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=40, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        tform, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(tform, src)

        # check error
        # mean_error = np.mean(distances)
        ss = src[:, np.newaxis, :].repeat(src.shape[1], axis=1)
        tt = dst[..., np.newaxis].repeat(dst.shape[1], axis=2)
        # chamfer dist
        mean_error = 0.5 * (
            (((ss - tt) ** 2).sum(0) ** 0.5).min(0).mean()
            + (((ss - tt) ** 2).sum(0) ** 0.5).min(0).mean()
        )
        # earth mover dist
        # scipy.optimize.linear_sum_assignment
        # import pdb;pdb.set_trace()
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    tform, _, _ = best_fit_transform(A, src[:m, :].T)
    # import pdb;pdb.set_trace()
    return tform, mean_error  # T, distances, i


def sample_pc(pc, num_samples=4000):
    if pc.shape[0] > num_samples:
        c_mask = np.zeros(pc.shape[0], dtype=int)
        c_mask[:num_samples] = 1
        np.random.shuffle(c_mask)
        masked_pc = pc[c_mask.nonzero()]
    else:
        masked_pc = np.pad(pc, ((0, num_samples - pc.shape[0]), (0, 0)), "wrap")
    return masked_pc
