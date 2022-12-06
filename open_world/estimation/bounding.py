from itertools import combinations

import numpy as np
from pybullet_tools.utils import (
    AABB,
    BLACK,
    INF,
    OOBB,
    Pose,
    aabb_from_points,
    add_segments,
    convex_centroid,
    convex_hull,
    draw_oobb,
    draw_pose,
    euler_from_quat,
    get_distance,
    invert,
    oobb_from_points,
    pose_from_tform,
    quat_from_matrix,
    quat_from_pose,
    tform_points,
)

from open_world.simulation.utils import interpolate_exterior


def get_o3d_aabb(points):
    import open3d

    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    o3d_aabb = pcd.get_axis_aligned_bounding_box()
    lower = o3d_aabb.get_center() - o3d_aabb.get_extent() / 2.0
    upper = o3d_aabb.get_center() + o3d_aabb.get_extent() / 2.0
    aabb = AABB(lower, upper)
    # draw_aabb(aabb)
    return aabb


def convex_hull_2d(points):
    return convex_hull(
        point[:2] for point in points
    ).vertices  # TODO: project onto surface


def get_surface_oobb(points, min_z):
    base_vertices2d = convex_hull_2d(points)
    base_centroid = np.append(
        convex_centroid(base_vertices2d), [min_z]
    )  # TODO: 3D mesh centroid
    base_exterior = interpolate_exterior(base_vertices2d)  # TODO: sample interior
    base_oobb = oobb_from_points(base_exterior)
    base_euler = euler_from_quat(quat_from_pose(base_oobb.pose))
    origin_pose = Pose(base_centroid, base_euler)
    points_origin = tform_points(invert(origin_pose), points)
    oobb = OOBB(aabb_from_points(points_origin), origin_pose)
    # draw_oobb(oobb)
    return oobb


def oriented_bounds_3D(points):
    from trimesh.bounds import oriented_bounds_2D

    # TODO: handle degenerate cases (try/except)
    min_z = np.min(points, axis=0)[2]
    max_z = np.max(points, axis=0)[2]
    points2d = [point[:2] for point in points]
    # print(len(points), np.min(points, axis=0), np.max(points, axis=0))
    # print(corners(points2d)) # Given a pair of axis aligned bounds, return all 8 corners of the bounding box.
    # TODO: scipy.spatial.qhull.QhullError: QH6022 qhull input error: 0'th dimension's new bounds [-0.5, 0.5]
    #  too wide for existing bounds [0.2, 0.2]
    tform2d, extent2d = oriented_bounds_2D(points2d)
    extent = np.append(extent2d, [max_z - min_z])
    tform = np.eye(4)
    tform[:2, :2] = tform2d[:2, :2]
    tform[:2, 3] = tform2d[:2, 2]
    tform[2, 3] = -(min_z + max_z) / 2.0
    return tform, extent


def get_trimesh_oobb(points, use_2d=True):
    # TODO: these seem to be slightly off
    from trimesh.bounds import oriented_bounds  # , minimum_cylinder, corners

    # from trimesh.nsphere import minimum_nsphere
    # cylinder_info = minimum_cylinder(points)
    # print(cylinder_info['radius'], cylinder_info['height'])
    # TODO: draw cylinder
    if use_2d:
        tform, extent = oriented_bounds_3D(points)
    else:
        tform, extent = oriented_bounds(points)
    lower = -extent / 2.0
    upper = +extent / 2.0
    oobb = OOBB(AABB(lower, upper), invert(pose_from_tform(tform)))
    return oobb


def get_o3d_oobb(points):
    # TODO: 2D version of this method
    import open3d

    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    o3d_oobb = pcd.get_oriented_bounding_box()
    lower = -o3d_oobb.extent / 2.0
    upper = +o3d_oobb.extent / 2.0
    pose = Pose(o3d_oobb.center, euler_from_quat(quat_from_matrix(o3d_oobb.R)))
    oobb = OOBB(AABB(lower, upper), pose)
    # draw_oobb(oobb)
    return oobb


def estimate_oobb(points, min_z=None, draw=False):
    if min_z is None:
        min_z = np.min(points, axis=0)[2]
    if draw:
        base_vertices_2d = convex_hull_2d(points)
        base_vertices = [np.append(vertex2d, [min_z]) for vertex2d in base_vertices_2d]
        add_segments(base_vertices, color=BLACK)
    # open3d.visualization.draw_geometries([pcd])
    # user_input()

    # TODO: best cylinder or box approximation
    # aabb = get_o3d_aabb(points)
    # oobb = get_o3d_oobb(points) # TODO: might still be a bug
    # oobb = get_surface_oobb(points, min_z)
    oobb = get_trimesh_oobb(points, use_2d=True)
    if draw:
        draw_pose(oobb.pose)
        draw_oobb(oobb)
    return oobb


##################################################


def fit_circle(*points):
    # https://www.geeksforgeeks.org/complex-numbers-in-python-set-1-introduction/
    assert len(points) == 3
    x, y, z = [complex(*point) for point in points]
    w = z - x
    w /= y - x
    c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    # print('(x%+.3f)^2+(y%+.3f)^2 = %.3f^2' % (c.real, c.imag, ))
    center = -np.array([c.real, c.imag])
    radius = abs(c + x)

    return center, radius


def min_circle(points):
    # from trimesh.nsphere import minimum_nsphere, fit_nsphere
    if len(points) < 3:
        return None
    best_center, best_radius = None, INF
    for triplet in combinations(points, r=3):  # random.sample(base_vertices_2d, k=3)
        center, radius = fit_circle(*triplet)
        if any(get_distance(center, point) > radius for point in points):
            continue
        if radius < best_radius:
            best_center, best_radius = center, radius
    return best_center, best_radius
