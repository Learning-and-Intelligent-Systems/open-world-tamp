import sys

import numpy as np
from pybullet_tools.utils import (
    INF,
    Mesh,
    apply_alpha,
    create_mesh,
    draw_oobb,
    draw_pose,
    invert,
    mesh_from_points,
    multiply,
    pixel_from_point,
    set_pose,
    tform_points,
)

from open_world.estimation.bounding import get_trimesh_oobb
from open_world.estimation.observation import aggregate_color
from open_world.simulation.lis import SC_PATH


def complete_shape(sc_network, points):  # deprecated
    import torch

    assert sc_network is not None
    # TODO: only works for atlas

    with torch.no_grad():
        points_arr = np.asarray(points)
        mean = points_arr.mean(0)
        points_shift = points_arr - mean  # zero mean
        scale_fac = (points_shift ** 2).sum(1).max() ** 0.5  # normalize
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
            else mesh_from_points(np.concatenate([submesh[0] for submesh in mesh_list]))
        )
    return obj_estimate


##################################################


def post_process(mesh, net_input, num_refine=10):
    # post processing. align the observed side of output mesh and input point cloud
    from auxiliary.ChamferDistancePytorch.chamfer_python import (
        distChamfer as Loss_chamfer,
    )

    for j in range(num_refine):
        mesh = mesh.detach().requires_grad_(True)
        mesh.retain_grad()
        chamfer1, chamfer2, _, _ = Loss_chamfer(
            net_input.transpose(2, 1), mesh.squeeze(1).transpose(2, 1)
        )
        # print(chamfer1.mean().item(),)
        chamfer1.mean().backward()
        grad = mesh.grad
        mesh = mesh - 10 * grad
    return mesh


def refine_shape(sc_network, points, num_refine=None, use_points=True, min_z=0.0):
    assert sc_network is not None
    import torch

    sys.path.append(SC_PATH)

    points_arr = np.asarray(points)
    mean = points_arr.mean(0)
    points_shift = points_arr - mean  # zero mean
    scale_fac = (points_shift ** 2).sum(1).max() ** 0.5  # normalize
    points_normed = (points_shift / scale_fac).transpose(1, 0)

    net_input = (
        torch.Tensor(points_normed).unsqueeze(0).to(sc_network.device)
    )  # 1(batch_size) x 3 x N
    mesh = sc_network.forward(net_input)  # 1 x 1 x 3 x N
    if num_refine:
        mesh = post_process(mesh, net_input, num_refine=num_refine)

    new_points = (mesh[0][0].detach().cpu().numpy() * scale_fac).transpose(1, 0) + mean
    new_points = [
        point for point in new_points if point[2] >= min_z
    ]  # TODO: could project instead
    if use_points:
        new_points = np.vstack([new_points, points])  # Doesn't change much in practice
    return Mesh(new_points, faces=None)


##################################################


def filter_visible(points, origin_pose, camera_image, instance=None, epsilon=5e-3):
    # TODO: record visible vertices to reason about visible faces during grasping
    # TODO: sample exterior of the mesh
    camera_pose, camera_matrix = camera_image[-2:]
    filtered_points = []
    camera_from_origin = multiply(invert(camera_pose), origin_pose)
    point_camera = tform_points(camera_from_origin, points)
    for idx, point_camera in enumerate(point_camera):
        pixel = pixel_from_point(camera_matrix, point_camera)
        if pixel is not None:
            depth = camera_image.depthPixels[pixel]  # TODO: median filter
            if np.isnan(depth):
                depth = INF
            x, y, z = point_camera
            if z >= (depth - epsilon):
                filtered_points.append(points[idx])
            # new_point = Point(x, y, min(depth, z)) # TODO: only project if the label is correct
            # filtered_points.append(tform_point(invert(camera_from_origin), new_point)) # TODO: need to clip at surface
        else:
            filtered_points.append(points[idx])
    if len(filtered_points) == len(point_camera):
        return points
    # handles = draw_points(filtered_points, color=GREEN) #, parent=obj_estimate))
    # wait_if_gui()
    return filtered_points


def inspect_mesh(labeled_points, alpha=0.5, draw=False, **kwargs):
    color = apply_alpha(aggregate_color(labeled_points), alpha=alpha)
    points = [point.point for point in labeled_points]
    oobb = get_trimesh_oobb(points, use_2d=True)
    if draw:
        draw_pose(oobb.pose) + draw_oobb(oobb)
    points_origin = tform_points(invert(oobb.pose), points)
    mesh = mesh_from_points(points_origin)
    # draw_mesh(mesh, color=color)
    body = create_mesh(mesh, under=True, color=color)
    set_pose(body, oobb.pose)
    return body
