import time
from collections import namedtuple

import numpy as np
import open3d
from trimesh.points import plane_fit, plane_transform, project_to_plane

import owt.pb_utils as pbu
import owt.utils as utils
from owt.estimation.bounding import get_trimesh_oobb
from owt.simulation.utils import Z_AXIS

Plane = namedtuple("Plane", ["normal", "origin"])
Surface = namedtuple("Surface", ["vertices", "pose"])


def z_plane(z=0.0):
    normal = Z_AXIS
    origin = z * normal
    return Plane(normal, origin)


def create_rectangular_surface(width, length):
    extents = np.array([width, length, 0]) / 2.0
    unit_corners = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]
    return [np.append(c, 0) * extents for c in unit_corners]


def rectangular_surface(width, length, pose):
    vertices = create_rectangular_surface(width, length)
    return Surface(vertices, pose)


def infinite_surface(pose):
    return rectangular_surface(width=np.inf, length=np.inf, pose=pose)


def plane_from_equation(plane_eqn):
    [a, b, c, d] = plane_eqn
    normal = pbu.get_unit_vector([a, b, c])
    origin = np.array([0, 0, -d / c])
    return Plane(normal, origin)


def equation_from_plane(plane):
    normal, origin = plane
    normal = pbu.get_unit_vector(normal)
    a, b, c = normal
    d = -np.dot(normal, origin)
    return [a, b, c, d]


def plane_from_pose(pose):
    origin = pbu.point_from_pose(pose)
    normal = np.array(pbu.tform_point(pose, Z_AXIS)) - np.array(origin)
    return Plane(normal, origin)


def point_plane_distance(plane, point, signed=True):
    plane_normal, plane_point = plane
    signed_distance = np.dot(plane_normal, np.array(point) - np.array(plane_point))
    if signed:
        return signed_distance
    return abs(signed_distance)


def line_plane_intersection(plane, ray_direction, ray_point, epsilon=1e-6):
    plane_normal, plane_point = plane
    inner_product = plane_normal.dot(ray_direction)
    if abs(inner_product) < epsilon:
        return None
    difference = ray_point - plane_point
    si = -plane_normal.dot(difference) / inner_product
    psi = difference + plane_point + si * ray_direction
    return psi


def project_plane(plane, point):
    # from trimesh.points import project_to_plane
    normal, _ = plane
    return np.array(point) - point_plane_distance(plane, point) * normal


def get_plane_quat(normal):
    plane = Plane(normal, np.zeros(3))
    normal, origin = plane
    tform = np.linalg.inv(plane_transform(origin, -normal))  # origin=None
    quat1 = pbu.quat_from_matrix(tform)
    pose1 = pbu.Pose(origin, euler=pbu.euler_from_quat(quat1))

    projection_world = project_plane(plane, Z_AXIS)
    projection = pbu.tform_point(pbu.invert(pose1), projection_world)
    yaw = pbu.get_yaw(projection[:2])
    quat2 = pbu.multiply_quats(quat1, pbu.quat_from_euler(pbu.Euler(yaw=yaw)))

    return quat2


def compute_inliers(plane, points, threshold=5e-2):
    from trimesh.points import point_plane_distance as point_plane_distances

    normal, origin = plane
    distances = np.absolute(point_plane_distances(points, normal, origin))
    inliers = [
        index for index, distance in enumerate(distances) if distance <= threshold
    ]
    return inliers


def ransac_estimate_plane(
    points, batch_size=3, num_iterations=250, axis=Z_AXIS, max_error=np.inf, **kwargs
):
    assert batch_size >= 3
    assert len(points) >= batch_size
    start_time = time.time()
    num_exceeding = 0
    best_plane = None
    best_inliers = []
    # TODO: could assume orientation and perform ransac on the height (or take the median)
    for i in range(num_iterations):
        batch = utils.safe_sample(points, k=batch_size)
        origin, normal = plane_fit(batch)
        if np.dot(normal, axis) < 0.0:
            normal *= -1
        if pbu.angle_between(normal, axis) > max_error:
            num_exceeding += 1
            continue
        plane = Plane(normal, origin)
        inliers = compute_inliers(plane, points, **kwargs)
        if len(inliers) > len(best_inliers):
            best_plane = plane
            best_inliers = inliers
    print(
        "Plane: {} | Exceeding: {} | Points: {} | Inliers: {} | Time: {:.3f} sec".format(
            best_plane,
            num_exceeding,
            len(points),
            len(best_inliers),
            pbu.elapsed_time(start_time),
        )
    )

    return best_plane, best_inliers


##################################################


def draw_surface(surface, **kwargs):
    plane_vertices, plane_pose = surface
    vertices_world = pbu.tform_points(plane_pose, plane_vertices)
    with pbu.LockRenderer():
        handles = pbu.draw_pose(plane_pose)
        handles.extend(pbu.add_segments(vertices_world, closed=True, **kwargs))
        for point in vertices_world:
            handles.extend(pbu.draw_point(point, color=pbu.RED))
    return handles


def create_surface(
    plane,
    points,
    max_distance=1e-2,
    min_area=0.0,
    origin_type="box",
    draw=False,
    **kwargs
):

    normal, origin = plane

    if len(points) < 3:
        return None
    points_plane, tform = project_to_plane(
        np.asarray(points),
        plane_normal=normal,
        plane_origin=origin,
        return_transform=True,
        return_planar=False,
    )  # origin influences position
    points_plane = [
        np.array([x, y]) for x, y, z in points_plane if abs(z) <= max_distance
    ]  # TODO: compute_inliers()
    if len(points_plane) < 3:
        return None
    if pbu.get_aabb_volume(pbu.aabb_from_points(points_plane)) <= min_area:
        return None
    # print(major_axis(points_plane)) # SVD
    vertices_plane = np.array(pbu.convex_hull(points_plane).vertices)
    area = pbu.convex_area(vertices_plane)
    if area <= min_area:
        return None
    n, d = vertices_plane.shape
    if d == 2:
        vertices_plane = np.hstack([vertices_plane, np.zeros([n, 1])])
    centroid_plane = np.append(pbu.convex_centroid(vertices_plane), [0.0])

    plane_pose = pbu.pose_from_tform(tform)

    if origin_type == "none":
        surface_pose = plane_pose
    elif origin_type == "box":
        oobb_plane = get_trimesh_oobb(vertices_plane, use_2d=True)
        oobb_world = pbu.tform_oobb(plane_pose, oobb_plane)
        surface_pose = oobb_world.pose
    elif origin_type == "centroid":
        centroid_world = pbu.tform_point(plane_pose, centroid_plane)
        tform = np.linalg.inv(plane_transform(origin, normal))
        surface_pose = pbu.Pose(
            centroid_world, pbu.euler_from_quat(pbu.quat_from_matrix(tform))
        )
    else:
        raise NotImplementedError(origin_type)

    vertices_world = pbu.tform_points(plane_pose, vertices_plane)
    vertices_surface = pbu.tform_points(pbu.invert(surface_pose), vertices_world)

    surface = Surface(vertices_surface, surface_pose)
    if draw:
        handles = draw_surface(surface)
        pbu.wait_if_gui()
        pbu.remove_handles(handles)
    return surface


def is_point_in_polygon(point, polygon):
    # TODO: aabb_contains_point
    sign = None
    for i in range(len(polygon)):
        v1, v2 = np.array(polygon[i - 1][:2]), np.array(polygon[i][:2])
        delta = v2 - v1
        normal = np.array([-delta[1], delta[0]])
        dist = normal.dot(point[:2] - v1)
        if i == 0:  # TODO: equality?
            sign = np.sign(dist)
        elif np.sign(dist) != sign:
            return False
    return True


def surface_point_filter(surface, labeled_points, min_z=1e-3):
    start_time = time.time()
    points_surface = pbu.tform_points(
        pbu.invert(surface.pose), [point.point for point in labeled_points]
    )
    surface_aabb = pbu.aabb2d_from_aabb(pbu.aabb_from_points(surface.vertices))
    filtered_points = [
        labeled_points[index]
        for index, point in enumerate(points_surface)
        if (min_z <= point[2])
        and pbu.aabb_contains_point(point[:2], surface_aabb)
        and is_point_in_polygon(point, surface.vertices)
    ]
    print(
        "Filter retained {}/{} in {:.3f} seconds".format(
            len(filtered_points), len(labeled_points), pbu.elapsed_time(start_time)
        )
    )
    return filtered_points


##################################################


def estimate_plane_eqns(points, num=3):

    time.time()
    centroid = np.average(points, axis=0)

    planes = []
    while len(points) >= num:
        pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
        plane_eqn, inlier_indices = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        normal, origin = plane_from_equation(plane_eqn)
        if point_plane_distance(Plane(normal, origin), centroid) > 0:
            normal *= -1
        plane_eqn = equation_from_plane(Plane(normal, origin))
        planes.append(Plane(normal, origin))
        points = [
            point for idx, point in enumerate(points) if idx not in inlier_indices
        ]
    return planes


def estimate_planes(points, **kwargs):
    planes = estimate_plane_eqns(points, **kwargs)
    handles = []
    with pbu.LockRenderer():
        for point in points:
            color = pbu.BLUE
            handles.extend(pbu.draw_point(point, color=color))

    surfaces = []
    new_points = []
    for plane in planes:
        surface = create_surface(plane, points)
        surfaces.append(surface)
        surface_vertices, surface_pose = surface
        new_points.extend(pbu.tform_points(surface_pose, surface_vertices))
        handles.extend(draw_surface(surface))
    pbu.wait_if_gui()
    return surfaces
