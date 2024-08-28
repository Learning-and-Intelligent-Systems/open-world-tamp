import sys

import numpy as np
import torch

import owt.pb_utils as pbu
from owt.estimation.bounding import get_trimesh_oobb
from owt.estimation.observation import aggregate_color
from owt.simulation.lis import SC_PATH


def complete_shape(sc_network, points):  # deprecated
    assert sc_network is not None
    # TODO: only works for atlas

    with torch.no_grad():
        points_arr = np.asarray(points)
        mean = points_arr.mean(0)
        points_shift = points_arr - mean  # zero mean
        scale_fac = (points_shift**2).sum(1).max() ** 0.5  # normalize
        points_normed = (points_shift / scale_fac).transpose(1, 0)
        net_input = (
            torch.Tensor(points_normed).unsqueeze(0).to(sc_network.device)
        )  # 1(batch_size) x 3 x N
        mesh_list = sc_network.generate_mesh(net_input)

        recon_list = []
        for submesh in mesh_list:
            vertices = submesh[0]
            vertices *= scale_fac
            vertices += mean
            recon_list.append(submesh)

        # TODO: smarter way to merge submesh
        # len(mesh_list) is always 1 for now(using SPHERE atlas)
        obj_estimate = (
            recon_list[0]
            if len(recon_list) == 1
            else pbu.mesh_from_points(
                np.concatenate([submesh[0] for submesh in mesh_list])
            )
        )
    return obj_estimate


##################################################


def refine_shape(sc_network, points, use_points=True, min_z=0.0, **kwargs):
    assert sc_network is not None

    sys.path.append(SC_PATH)

    points_arr = np.asarray(points)
    mean = points_arr.mean(0)
    points_shift = points_arr - mean  # zero mean
    scale_fac = (points_shift**2).sum(1).max() ** 0.5  # normalize
    points_normed = (points_shift / scale_fac).transpose(1, 0)

    net_input = torch.Tensor(points_normed).unsqueeze(0).to(sc_network.device)
    mesh = sc_network.forward(net_input)  # 1 x 1 x 3 x N

    new_points = (mesh[0][0].detach().cpu().numpy() * scale_fac).transpose(1, 0) + mean
    new_points = [point for point in new_points if point[2] >= min_z]
    if use_points:
        new_points = np.vstack([new_points, points])  # Doesn't change much in practice
    return pbu.Mesh(new_points, faces=None)


##################################################


def filter_visible(points, origin_pose, camera_image, instance=None, epsilon=5e-3):
    camera_pose, camera_matrix = camera_image[-2:]
    filtered_points = []
    camera_from_origin = pbu.multiply(pbu.invert(camera_pose), origin_pose)
    point_camera = pbu.tform_points(camera_from_origin, points)
    for idx, point_camera in enumerate(point_camera):
        pixel = pbu.pixel_from_point(camera_matrix, point_camera)
        if pixel is not None:
            depth = camera_image.depthPixels[pixel]  # TODO: median filter
            if np.isnan(depth):
                depth = np.inf
            x, y, z = point_camera
            if z >= (depth - epsilon):
                filtered_points.append(points[idx])
        else:
            filtered_points.append(points[idx])
    if len(filtered_points) == len(point_camera):
        return points
    return filtered_points


def inspect_mesh(labeled_points, alpha=0.5, draw=False, **kwargs):
    color = pbu.apply_alpha(aggregate_color(labeled_points), alpha=alpha)
    points = [point.point for point in labeled_points]
    oobb = get_trimesh_oobb(points, use_2d=True)
    if draw:
        pbu.draw_pose(oobb.pose) + pbu.draw_oobb(oobb)
    points_origin = pbu.tform_points(pbu.invert(oobb.pose), points)
    mesh = pbu.mesh_from_points(points_origin)
    body = pbu.create_mesh(mesh, under=True, color=color)
    pbu.set_pose(body, oobb.pose)
    return body