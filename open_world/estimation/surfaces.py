import time
from collections import namedtuple

import numpy as np
from pybullet_tools.utils import (
    BLUE,
    INF,
    RED,
    Euler,
    LockRenderer,
    Pose,
    aabb2d_from_aabb,
    aabb_contains_point,
    aabb_from_points,
    add_segments,
    angle_between,
    convex_area,
    convex_centroid,
    convex_hull,
    create_rectangular_surface,
    draw_point,
    draw_pose,
    elapsed_time,
    euler_from_quat,
    get_aabb_volume,
    get_unit_vector,
    get_yaw,
    invert,
    is_point_in_polygon,
    multiply_quats,
    point_from_pose,
    pose_from_tform,
    quat_from_euler,
    quat_from_matrix,
    remove_handles,
    safe_sample,
    tform_oobb,
    tform_point,
    tform_points,
    wait_if_gui,
)

from open_world.estimation.bounding import get_trimesh_oobb
from open_world.simulation.utils import Z_AXIS

# TODO: merge into pybullet_tools/utils.py
Plane = namedtuple("Plane", ["normal", "origin"])
# Plane = namedtuple('Plane', ['a', 'b', 'c', 'd'])
Surface = namedtuple("Surface", ["vertices", "pose"])


def z_plane(z=0.0):
    normal = Z_AXIS
    origin = z * normal
    return Plane(normal, origin)


def rectangular_surface(width, length, pose):
    vertices = create_rectangular_surface(width, length)
    return Surface(vertices, pose)


def infinite_surface(pose):
    return rectangular_surface(width=INF, length=INF, pose=pose)


def plane_from_equation(plane_eqn):
    [a, b, c, d] = plane_eqn
    normal = get_unit_vector([a, b, c])
    origin = np.array([0, 0, -d / c])
    return Plane(normal, origin)


def equation_from_plane(plane):
    normal, origin = plane
    normal = get_unit_vector(normal)
    a, b, c = normal
    d = -np.dot(normal, origin)
    return [a, b, c, d]


def plane_from_pose(pose):
    origin = point_from_pose(pose)
    normal = np.array(tform_point(pose, Z_AXIS)) - np.array(origin)  # get_unit_vector
    return Plane(normal, origin)


def point_plane_distance(plane, point, signed=True):
    # TODO: reorder name
    plane_normal, plane_point = plane
    # from trimesh.points import point_plane_distance # NOTE: the output is signed
    signed_distance = np.dot(plane_normal, np.array(point) - np.array(plane_point))
    if signed:
        return signed_distance
    return abs(signed_distance)


def line_plane_intersection(plane, ray_direction, ray_point, epsilon=1e-6):
    plane_normal, plane_point = plane
    # from trimesh.intersections import plane_lines
    # https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
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
    from trimesh.points import plane_transform

    # from trimesh.geometry import align_vectors
    # transform that will move that plane to be coplanar with the XY plane
    plane = Plane(normal, np.zeros(3))
    normal, origin = plane
    tform = np.linalg.inv(plane_transform(origin, -normal))  # origin=None
    quat1 = quat_from_matrix(tform)
    pose1 = Pose(origin, euler=euler_from_quat(quat1))

    projection_world = project_plane(plane, Z_AXIS)
    projection = tform_point(invert(pose1), projection_world)
    yaw = get_yaw(projection[:2])
    quat2 = multiply_quats(quat1, quat_from_euler(Euler(yaw=yaw)))

    return quat2


def compute_inliers(plane, points, threshold=5e-2):
    from trimesh.points import point_plane_distance as point_plane_distances

    normal, origin = plane
    distances = np.absolute(point_plane_distances(points, normal, origin))
    # distances = [abs(point_plane_distance(plane, point)) for point in points]
    # TODO: select_inliers function
    inliers = [
        index for index, distance in enumerate(distances) if distance <= threshold
    ]
    return inliers


def ransac_estimate_plane(
    points, batch_size=3, num_iterations=250, axis=Z_AXIS, max_error=INF, **kwargs
):
    # https://github.com/intel-isl/Open3D/blob/1728fc12934561321623c8798ef3b83059137321/cpp/open3d/geometry/PointCloudSegmentation.cpp#L135
    # https://github.com/daavoo/pyntcloud/blob/7c5daf825bcb84e6f750a6d6fad3a2626c4ea5ec/pyntcloud/utils/array.py#L53
    # https://github.com/daavoo/pyntcloud/blob/7c5daf825bcb84e6f750a6d6fad3a2626c4ea5ec/pyntcloud/geometry/models/plane.py#L25
    # https://github.com/ajhynes7/scikit-spatial/blob/16140440e6152a98587fabfc70f0003520192453/skspatial/objects/plane.py#L551
    # https://github.com/falcondai/py-ransac/blob/master/plane_fitting.py
    # https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    # https://gist.github.com/RustingSword/e22a11e1d391f2ab1f2c
    # https://github.com/caelan/PyR2/blob/99718793ebe77c2913cae0f6a6c379c5362aaf58/PyR2/shapes.pyx#L808
    # from numpy.linalg import eig, svd, lstsq # TODO: SVD vs eigen
    from trimesh.points import plane_fit

    # from open_world.retired import estimate_pcl_plane
    assert batch_size >= 3
    assert len(points) >= batch_size
    start_time = time.time()
    num_exceeding = 0
    best_plane = None
    best_inliers = []
    # TODO: could assume orientation and perform ransac on the height (or take the median)
    for i in range(num_iterations):
        batch = safe_sample(points, k=batch_size)
        origin, normal = plane_fit(batch)
        if np.dot(normal, axis) < 0.0:
            normal *= -1
        if angle_between(normal, axis) > max_error:
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
            elapsed_time(start_time),
        )
    )

    return best_plane, best_inliers


##################################################


def draw_surface(surface, **kwargs):
    plane_vertices, plane_pose = surface
    vertices_world = tform_points(plane_pose, plane_vertices)
    with LockRenderer():
        handles = draw_pose(plane_pose)
        handles.extend(add_segments(vertices_world, closed=True, **kwargs))
        for point in vertices_world:
            handles.extend(draw_point(point, color=RED))
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
    from trimesh.points import plane_transform, project_to_plane  # major_axis

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
    if get_aabb_volume(aabb_from_points(points_plane)) <= min_area:
        return None
    # print(major_axis(points_plane)) # SVD
    vertices_plane = np.array(convex_hull(points_plane).vertices)
    area = convex_area(vertices_plane)
    if area <= min_area:
        return None
    n, d = vertices_plane.shape
    if d == 2:
        vertices_plane = np.hstack([vertices_plane, np.zeros([n, 1])])
    centroid_plane = np.append(convex_centroid(vertices_plane), [0.0])

    plane_pose = pose_from_tform(tform)

    if origin_type == "none":
        surface_pose = plane_pose
    elif origin_type == "box":
        oobb_plane = get_trimesh_oobb(vertices_plane, use_2d=True)
        # print(oobb_plane)
        oobb_world = tform_oobb(plane_pose, oobb_plane)
        # draw_oobb(oobb_world, origin=True)
        # wait_if_gui()
        surface_pose = oobb_world.pose
    elif origin_type == "centroid":
        centroid_world = tform_point(plane_pose, centroid_plane)
        tform = np.linalg.inv(
            plane_transform(origin, normal)
        )  # transform that will move that plane to be coplanar with the XY plane
        surface_pose = Pose(centroid_world, euler_from_quat(quat_from_matrix(tform)))
    else:
        raise NotImplementedError(origin_type)

    vertices_world = tform_points(plane_pose, vertices_plane)
    vertices_surface = tform_points(invert(surface_pose), vertices_world)
    # TODO: intersection of planes for the hull

    surface = Surface(vertices_surface, surface_pose)
    if draw:
        handles = draw_surface(surface)
        wait_if_gui()
        remove_handles(handles)
    return surface


def surface_point_filter(surface, labeled_points, min_z=1e-2):
    # from pybullet_tools.utils import is_point_on_surface
    # from trimesh.path.polygons import projected
    # import trimesh
    # trimesh.base.Trimesh.contains
    # TODO: use trimesh functions
    start_time = time.time()
    points_surface = tform_points(
        invert(surface.pose), [point.point for point in labeled_points]
    )
    surface_aabb = aabb2d_from_aabb(aabb_from_points(surface.vertices))
    filtered_points = [
        labeled_points[index]
        for index, point in enumerate(points_surface)
        if (min_z <= point[2])
        and aabb_contains_point(point[:2], surface_aabb)
        and is_point_in_polygon(point, surface.vertices)
    ]
    print(
        "Filter retained {}/{} in {:.3f} seconds".format(
            len(filtered_points), len(labeled_points), elapsed_time(start_time)
        )
    )
    return filtered_points


##################################################


def estimate_plane_eqns(points, num=3):
    # TODO: orientation filter
    import open3d

    time.time()
    # import open3d as o3d
    centroid = np.average(points, axis=0)

    # pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    # pcd.estimate_normals(
    #     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    # length = 0.05
    # handles = []
    # with LockRenderer():
    #     for point, normal in zip(pcd.points, pcd.normals):
    #         draw_point(point)
    #         normal = get_unit_vector(normal)
    #         plane = Plane(normal, point)
    #         if point_plane_distance(plane, centroid) > 0:
    #             normal *= -1
    #         handles.append(add_line(point, point + length*normal, color=RED))
    # # TODO: only return for close enough points
    # wait_if_gui()

    planes = []
    while len(points) >= num:
        pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True) # TODO: segfault
        plane_eqn, inlier_indices = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        [a, b, c, d] = plane_eqn
        # print(f'Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z = {-d:.3f} | Inliers: {len(inlier_indices)}')
        print(
            "Plane equation: {:.3f}x + {:.3f}y + {:.3f}z = {:.3f} | Inliers: {}".format(
                a, b, c, -d, len(inlier_indices)
            )
        )

        normal, origin = plane_from_equation(plane_eqn)
        if point_plane_distance(Plane(normal, origin), centroid) > 0:
            normal *= -1
        plane_eqn = equation_from_plane(
            Plane(normal, origin)
        )  # Can expand the point a little bit
        # planes.append(plane_eqn)
        planes.append(Plane(normal, origin))
        # inlier_cloud = pcd.select_by_index(inlier_indices)
        # inlier_points = [points[idx] for idx in inlier_indices]
        # orient_normals_consistent_tangent_plane
        points = [
            point for idx, point in enumerate(points) if idx not in inlier_indices
        ]
        # TODO: include prior points when computing fit quality
    # print(len(planes), elapsed_time(start_time))
    return planes


def estimate_planes(points, **kwargs):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html#scipy.spatial.HalfspaceIntersection
    # from scipy.spatial import HalfspaceIntersection
    # from trimesh.intersections import slice_faces_plane
    planes = estimate_plane_eqns(points, **kwargs)

    # TODO: add the surface plane
    # vertices, faces = mesh_from_points(points)
    # handles = draw_mesh(Mesh(vertices, faces))
    # wait_if_gui()
    # remove_handles(handles)
    # for plane in planes:
    #     plane_normal, plane_origin = plane
    #     plane_normal = -get_unit_vector(plane_normal)
    #     plane_origin = plane_origin + 1e-2*plane_normal
    #     vertices, faces = slice_faces_plane(np.array(vertices), np.array(faces), plane_normal, plane_origin)
    #     handles = draw_mesh(Mesh(vertices.tolist(), faces.tolist()))
    #     wait_if_gui()
    #     remove_handles(handles)

    handles = []
    with LockRenderer():
        for point in points:
            color = BLUE
            # if all(point_plane_distance(plane, point) <= 0 for plane in planes):
            #     color = RED
            handles.extend(draw_point(point, color=color))

    surfaces = []
    new_points = []
    for plane in planes:
        surface = create_surface(plane, points)
        surfaces.append(surface)
        surface_vertices, surface_pose = surface
        new_points.extend(tform_points(surface_pose, surface_vertices))
        handles.extend(draw_surface(surface))
        # TODO: fit to multiple planes
        # TODO: assign points to only one plane

        # with LockRenderer():
        #     indices = list(range(len(points)))
        #     for idx in set(indices) - set(inlier_indices):
        #        handles.extend(draw_point(points[idx], color=BLACK))

        # wait_if_gui()
        # remove_handles(handles)
        # points = [point for idx, point in enumerate(points) if idx not in inlier_indices]

    # obj_mesh = mesh_from_points(new_points)
    # obj_estimate = create_mesh(obj_mesh, under=True) #, color=mean_color)
    wait_if_gui()
    return surfaces
