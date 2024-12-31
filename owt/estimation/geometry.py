import math
import os
import time

import numpy as np
import trimesh
from trimesh.intersections import slice_faces_plane

import owt.pb_utils as pbu
from owt.estimation.bounding import convex_hull_2d, estimate_oobb, min_circle
from owt.estimation.completion import filter_visible, refine_shape
from owt.estimation.concave import (concave_mesh, create_concave_mesh,
                                    create_mesh)
from owt.estimation.observation import (aggregate_color, draw_points,
                                        tform_labeled_points)
from owt.estimation.surfaces import Plane, point_plane_distance
from owt.planning.grasping import mesh_from_obj
from owt.simulation.entities import BOWL
from owt.simulation.utils import Z_AXIS

AUGMENT_BOWLS = True
VISUALIZE_COLLISION = True
OBJ_MESH_CACHE = {}


def vertices_from_rigid(body, link=pbu.BASE_LINK):
    try:
        vertices = pbu.vertices_from_link(body, link)
    except RuntimeError:
        info = pbu.get_model_info(body)
        assert info is not None
        _, ext = os.path.splitext(info.path)
        if ext == ".obj":
            if info.path not in OBJ_MESH_CACHE:
                OBJ_MESH_CACHE[info.path] = pbu.read_obj(info.path, decompose=False)
            mesh = OBJ_MESH_CACHE[info.path]
            vertices = mesh.vertices
        else:
            raise NotImplementedError(ext)
    return vertices


def approximate_as_prism(body, body_pose=None, **kwargs):
    if body_pose is None:
        body_pose = pbu.unit_pose()
    vertices = pbu.tform_points(body_pose, vertices_from_rigid(body, **kwargs))
    aabb = pbu.aabb_from_points(vertices)
    return pbu.get_aabb_center(aabb), pbu.get_aabb_extent(aabb)


def hull_ransac(points, min_points=10, threshold=5e-3, draw=False):
    start_time = time.time()
    hull = pbu.Mesh(*map(np.array, pbu.convex_hull(points)))
    centroid = np.average(hull.vertices, axis=0)
    if draw:
        with pbu.LockRenderer():
            pbu.draw_mesh(hull)
    planes = []
    while len(points) >= min_points:
        best_plane, best_indices = None, []
        for i, face in enumerate(hull.faces):
            v1, v2, v3 = hull.vertices[face]
            normal = pbu.get_mesh_normal(hull.vertices[face], centroid)
            plane = Plane(normal, v1)
            indices = [
                index
                for index, point in enumerate(points)
                if abs(point_plane_distance(plane, point)) <= threshold
            ]  # TODO: cache
            if len(indices) > len(best_indices):
                best_plane, best_indices = plane, indices
        if len(best_indices) < min_points:
            break
        print(
            "{} | {} | {} | {:.3f}".format(
                len(planes), best_plane, len(best_indices), pbu.elapsed_time(start_time)
            )
        )
        if draw:
            handles = draw_points(
                [points[index] for index in best_indices], color=pbu.GREEN
            )
            pbu.wait_if_gui()
            pbu.remove_handles(handles)
        planes.append(best_plane)
        points = [
            point for index, point in enumerate(points) if index not in best_indices
        ]
    return planes


def cloud_from_depth(camera_matrix, depth, max_depth=10.0, top_left_origin=False):
    height, width = depth.shape
    xmap = np.array(
        [[i for i in range(width)] for _ in range(height)]
    )  # 0 ~ width. hxw
    if top_left_origin:
        ymap = np.array(
            [[j for _ in range(width)] for j in range(height)]
        )  # 0 ~ height. hxw
    else:
        ymap = np.array(
            [[height - j for _ in range(width)] for j in range(height)]
        )  # 0 ~ height. hxw
    homogeneous_coord = np.concatenate(
        [xmap.reshape(1, -1), ymap.reshape(1, -1), np.ones((1, height * width))]
    )  # 3 x (hw)
    rays = np.linalg.inv(camera_matrix).dot(homogeneous_coord)
    point_cloud = depth.reshape(1, height * width) * rays
    point_cloud = point_cloud.transpose(1, 0).reshape(height, width, 3)

    # Filter max depth
    point_cloud[point_cloud[:, :, 2] > max_depth] = 0
    return point_cloud


def trim_mesh(submesh, **kwargs):
    planes = [
        Plane(Z_AXIS, pbu.Point(z=np.min(submesh.vertices, axis=0)[2])),
    ]
    for plane in planes:
        normal, point = plane
        plane = Plane(normal, point)
        vertices, faces = map(np.array, submesh)
        submesh = pbu.Mesh(*slice_faces_plane(vertices, faces, *plane))

    print("Vertices: {} | Faces: {}".format(len(submesh.vertices), len(submesh.faces)))
    return submesh


##################################################


def project_base_points(points, min_z=0.0, max_z=np.inf):
    return points + [
        np.append(point[:2], [min_z]) for point in points if point[2] <= max_z
    ]


def project_points(points, min_z=0.0, resolution=1e-2):
    projected_points = []
    for point in points:
        base_point = np.append(point[:2], [min_z])
        num_steps = max(
            2, int(1 + math.ceil(pbu.get_distance(point, base_point) / resolution))
        )
        projected_points.extend(pbu.interpolate(point, base_point, num_steps=num_steps))
    return projected_points


def trimesh_from_body(body):
    mesh = mesh_from_obj(body)
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)


##################################################


def estimate_mesh(
    labeled_points,
    min_z=0.0,
    min_volume=1e-3**3,
    min_area=1e-3**2,
    min_height=2e-2,
    max_aspect=0.2,
    sc_network=None,
    use_geom=False,
    use_points=True,
    use_hull=True,
    project_base=True,
    concave=False,
    use_image=True,
    camera_image=None,
    min_points=30,
    **kwargs
):
    category, instance = labeled_points[0].label

    labeled_points = [
        lp for lp in labeled_points if lp.point[2] >= min_z
    ]  # TODO: could project instead of filtering

    points = [point.point for point in labeled_points]
    aabb = pbu.aabb_from_points(points)

    if any([(aabb.upper[i] - aabb.lower[i]) > max_aspect for i in range(3)]):
        return None
    if (
        (len(points) < 3)
        or (aabb.upper[2] < min_height)
        or (pbu.get_aabb_volume(aabb) < min_volume)
        or (pbu.get_aabb_area(aabb) < min_area)
    ):
        print("[Warning] skipping mesh: aabb smaller than min volume")
        return None
    obj_oobb = estimate_oobb(points)
    if (pbu.get_aabb_volume(obj_oobb.aabb) < min_volume) or (
        pbu.get_aabb_area(obj_oobb.aabb) < min_area
    ):
        print("[Warning] skipping mesh: oobb smaller than min volume")
        return None
    origin_pose = obj_oobb.pose  # TODO: adjust pose to be the base
    color = pbu.apply_alpha(pbu.RGBA(*aggregate_color(labeled_points)), alpha=0.75)

    base_vertices_2d = convex_hull_2d(points)
    if pbu.convex_area(base_vertices_2d) < min_area:
        print("[Warning] skipping mesh: base vertices aabb smaller than min area")
        return None

    if AUGMENT_BOWLS and (category == BOWL):
        concave = False
        center, radius = min_circle(base_vertices_2d)
        points.extend(
            pbu.get_circle_vertices(
                np.append(center, [np.max(points, axis=0)[2]]),
                0.9 * radius,
                n=int(math.ceil(360.0 / 5)),
            )
        )

    points_origin = pbu.tform_points(pbu.invert(origin_pose), points)
    base_points = project_points(points, min_z=min_z)
    base_origin = pbu.tform_points(pbu.invert(origin_pose), base_points)
    base_origin = filter_visible(base_origin + points_origin, origin_pose, camera_image)

    if use_geom or (sc_network is None):
        merged_origin = base_origin if project_base else points_origin
        obj_mesh = (
            concave_mesh(merged_origin)
            if concave
            else pbu.mesh_from_points(merged_origin)
        )
        if obj_mesh is None:
            return None
    else:
        obj_mesh = refine_shape(
            sc_network, points_origin, use_points=use_points, min_z=min_z, **kwargs
        )
        if use_image and (camera_image is not None):
            obj_mesh = pbu.Mesh(
                filter_visible(
                    obj_mesh.vertices, origin_pose, camera_image, instance=instance
                ),
                faces=None,
            )
            if len(obj_mesh.vertices) < min_points:
                return None

        if use_hull or (obj_mesh.faces is None):
            merged_origin = obj_mesh.vertices
            if use_points:
                merged_origin = np.vstack(
                    [merged_origin, base_origin if project_base else points_origin]
                )
            obj_mesh = (
                concave_mesh(merged_origin)
                if concave
                else pbu.mesh_from_points(merged_origin)
            )

        if len(obj_mesh.vertices) >= 3:
            obj_mesh = trim_mesh(obj_mesh)

    if concave:
        obj_estimate = create_concave_mesh(
            obj_mesh,
            under=False,
            color=None if VISUALIZE_COLLISION else color,
            **kwargs
        )
    else:
        obj_estimate = create_mesh(
            obj_mesh, under=True, color=None if VISUALIZE_COLLISION else color, **kwargs
        )
    pbu.set_pose(obj_estimate, origin_pose, **kwargs)
    return obj_estimate


def estimate_surface_mesh(
    labeled_points, surface_pose=None, camera_image: pbu.CameraImage = None, **kwargs
):
    if surface_pose is None:
        min_z = min(lp.point[2] for lp in labeled_points)
        surface_pose = pbu.Pose(pbu.Point(z=min_z))
    if camera_image is not None:
        camera_pose = pbu.multiply(pbu.invert(surface_pose), camera_image.camera_pose)
        camera_image = pbu.CameraImage(
            camera_image.rgbPixels,
            camera_image.depthPixels,
            camera_image.segmentationMaskBuffer,
            camera_pose,
            camera_image.camera_matrix,
        )

    labeled_cluster = tform_labeled_points(pbu.invert(surface_pose), labeled_points)
    body = estimate_mesh(labeled_cluster, camera_image=camera_image, **kwargs)
    if body is None:
        return body
    pbu.set_pose(
        body, pbu.multiply(surface_pose, pbu.get_pose(body, **kwargs)), **kwargs
    )
    return body
