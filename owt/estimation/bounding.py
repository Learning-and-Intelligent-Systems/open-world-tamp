from itertools import combinations

import numpy as np
import open3d
from trimesh.bounds import oriented_bounds

import owt.pb_utils as pbu
from owt.simulation.utils import interpolate_exterior


def get_o3d_aabb(points):
    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    o3d_aabb = pcd.get_axis_aligned_bounding_box()
    lower = o3d_aabb.get_center() - o3d_aabb.get_extent() / 2.0
    upper = o3d_aabb.get_center() + o3d_aabb.get_extent() / 2.0
    aabb = pbu.AABB(lower, upper)
    return aabb


def convex_hull_2d(points):
    return pbu.convex_hull(point[:2] for point in points).vertices


def get_surface_oobb(points, min_z):
    base_vertices2d = convex_hull_2d(points)
    base_centroid = np.append(pbu.convex_centroid(base_vertices2d), [min_z])
    base_exterior = interpolate_exterior(base_vertices2d)
    base_oobb = pbu.oobb_from_points(base_exterior)
    base_euler = pbu.euler_from_quat(pbu.quat_from_pose(base_oobb.pose))
    origin_pose = pbu.Pose(base_centroid, base_euler)
    points_origin = pbu.tform_points(pbu.invert(origin_pose), points)
    oobb = pbu.OOBB(pbu.aabb_from_points(points_origin), origin_pose)
    return oobb


def oriented_bounds_3D(points):
    from trimesh.bounds import oriented_bounds_2D

    min_z = np.min(points, axis=0)[2]
    max_z = np.max(points, axis=0)[2]
    points2d = [point[:2] for point in points]
    tform2d, extent2d = oriented_bounds_2D(points2d)
    extent = np.append(extent2d, [max_z - min_z])
    tform = np.eye(4)
    tform[:2, :2] = tform2d[:2, :2]
    tform[:2, 3] = tform2d[:2, 2]
    tform[2, 3] = -(min_z + max_z) / 2.0
    return tform, extent


def get_trimesh_oobb(points, use_2d=True):
    if use_2d:
        tform, extent = oriented_bounds_3D(points)
    else:
        tform, extent = oriented_bounds(points)
    lower = -extent / 2.0
    upper = +extent / 2.0
    oobb = pbu.OOBB(pbu.AABB(lower, upper), pbu.invert(pbu.pose_from_tform(tform)))
    return oobb


def get_o3d_oobb(points):
    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    o3d_oobb = pcd.get_oriented_bounding_box()
    lower = -o3d_oobb.extent / 2.0
    upper = +o3d_oobb.extent / 2.0
    pose = pbu.Pose(
        o3d_oobb.center, pbu.euler_from_quat(pbu.quat_from_matrix(o3d_oobb.R))
    )
    oobb = pbu.OOBB(pbu.AABB(lower, upper), pose)
    return oobb


def estimate_oobb(points, min_z=None, draw=False):
    if min_z is None:
        min_z = np.min(points, axis=0)[2]
    if draw:
        base_vertices_2d = convex_hull_2d(points)
        base_vertices = [np.append(vertex2d, [min_z]) for vertex2d in base_vertices_2d]
        pbu.add_segments(base_vertices, color=pbu.BLACK)
    oobb = get_trimesh_oobb(points, use_2d=True)
    if draw:
        pbu.draw_pose(oobb.pose)
        pbu.draw_oobb(oobb)
    return oobb


def fit_circle(*points):
    assert len(points) == 3
    x, y, z = [complex(*point) for point in points]
    w = z - x
    w /= y - x
    c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    center = -np.array([c.real, c.imag])
    radius = abs(c + x)

    return center, radius


def min_circle(points):
    if len(points) < 3:
        return None
    best_center, best_radius = None, np.inf
    for triplet in combinations(points, r=3):
        center, radius = fit_circle(*triplet)
        if any(pbu.get_distance(center, point) > radius for point in points):
            continue
        if radius < best_radius:
            best_center, best_radius = center, radius
    return best_center, best_radius
