import math
import os

import numpy as np
from pybullet_tools.utils import (
    INF,
    OBJ_MESH_CACHE,
    CameraImage,
    Mesh,
    Point,
    Pose,
    aabb_from_points,
    apply_alpha,
    convex_area,
    create_mesh,
    get_aabb_area,
    get_aabb_volume,
    get_circle_vertices,
    get_distance,
    get_model_info,
    get_pose,
    interpolate,
    invert,
    mesh_from_points,
    multiply,
    read_obj,
    set_pose,
    tform_points,
)

from open_world.estimation.bounding import convex_hull_2d, estimate_oobb, min_circle
from open_world.estimation.completion import filter_visible, refine_shape
from open_world.estimation.concave import concave_mesh, create_concave_mesh
from open_world.estimation.observation import aggregate_color, tform_labeled_points
from open_world.estimation.surfaces import Plane
from open_world.planning.grasping import mesh_from_obj
from open_world.retired.scrap import hull_ransac
from open_world.simulation.entities import BOWL
from open_world.simulation.utils import Z_AXIS

AUGMENT_BOWLS = True
VISUALIZE_COLLISION = True


def cloud_from_depth(camera_matrix, depth, max_depth=10.0, top_left_origin=False):
    # width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
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


def trim_mesh(submesh, trim=False):
    from trimesh.intersections import slice_faces_plane

    # import open3d
    # pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    # mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, open3d.utility.DoubleVector(radii))
    # hull = Mesh(mesh.vertices, mesh.triangles)

    planes = [
        Plane(Z_AXIS, Point(z=np.min(submesh.vertices, axis=0)[2])),
    ]
    if trim:
        # TODO: points is the vertices of the convex hull
        planes.extend(hull_ransac(points))
    # TODO: separate into standalone method
    for plane in planes:
        normal, point = plane
        plane = Plane(normal, point)  # + 1e-3*np.array(normal))
        vertices, faces = map(np.array, submesh)
        submesh = Mesh(*slice_faces_plane(vertices, faces, *plane))
        # with LockRenderer():
        #     handles = draw_mesh(submesh)
        # wait_if_gui()
        # remove_handles(handles)
    print("Vertices: {} | Faces: {}".format(len(submesh.vertices), len(submesh.faces)))
    return submesh


##################################################


def project_base_points(points, min_z=0.0, max_z=INF):
    return points + [
        np.append(point[:2], [min_z]) for point in points if point[2] <= max_z
    ]


def project_points(points, min_z=0.0, resolution=1e-2):
    projected_points = []
    for point in points:
        base_point = np.append(point[:2], [min_z])
        num_steps = max(
            2, int(1 + math.ceil(get_distance(point, base_point) / resolution))
        )
        projected_points.extend(interpolate(point, base_point, num_steps=num_steps))
    return projected_points


def trimesh_from_body(body):  # TODO make a property of obj
    # from open_world.planning.grasping import mesh_from_obj
    import trimesh

    info = get_model_info(body)
    if info is None:
        mesh = mesh_from_obj(body)
    else:
        assert info is not None
        _, ext = os.path.splitext(info.path)

        if ext == ".obj":
            if info.path not in OBJ_MESH_CACHE:
                OBJ_MESH_CACHE[info.path] = read_obj(info.path, decompose=False)
            mesh = OBJ_MESH_CACHE[info.path]
        else:
            raise RuntimeError()  # TODO(xiaolin): mesh might not be defined
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)


##################################################


def estimate_mesh(
    labeled_points,
    min_z=0.0,
    min_volume=1e-3 ** 3,
    min_area=1e-3 ** 2,
    min_height=2e-2,
    max_aspect = 0.2,
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
    # print("Estimating Mesh!")

    # concave &= has_open3d()
    category, instance = labeled_points[0].label

    labeled_points = [
        lp for lp in labeled_points if lp.point[2] >= min_z
    ]  # TODO: could project instead of filtering
    points = [point.point for point in labeled_points]
    # planes = estimate_planes(points)

    # COM: get_com_pose
    aabb = aabb_from_points(points)
    # print(aabb)
    # print("Height: {}--{}".format(aabb.upper[2], min_height))
    # print("Volume: {}--{}".format(get_aabb_volume(aabb), min_volume))
    # print("Area: {}--{}".format(get_aabb_area(aabb), min_area))
    
    if(any([(aabb.upper[i]-aabb.lower[i]) > \
        max_aspect for i in range(3)])):
        return None
    if (
        (len(points) < 3)
        or (aabb.upper[2] < min_height)
        or (get_aabb_volume(aabb) < min_volume)
        or (get_aabb_area(aabb) < min_area)
    ):
        print("aabb smaller than min volume")
        return None
    obj_oobb = estimate_oobb(points)  # , min_z=min_z)
    if (get_aabb_volume(obj_oobb.aabb) < min_volume) or (
        get_aabb_area(obj_oobb.aabb) < min_area
    ):
        print("oobb smaller than min volume")
        return None
    origin_pose = obj_oobb.pose  # TODO: adjust pose to be the base
    color = apply_alpha(
        aggregate_color(labeled_points), alpha=0.75
    )  # transparency=0.75 | 1.0
    # draw_oobb(obj_oobb, color=color)

    base_vertices_2d = convex_hull_2d(points)
    if convex_area(base_vertices_2d) < min_area:
        print("base vertices aabb smaller than min area")
        return None

    if AUGMENT_BOWLS and (category == BOWL):
        concave = False
        # base_vertices_3d = [np.append(vertex, [1e-3]) for vertex in base_vertices_2d]
        # draw_points(base_vertices_3d)
        # add_segments(base_vertices_3d)
        center, radius = min_circle(base_vertices_2d)
        # draw_circle(np.append(center, [1e-3]), radius, color=GREEN)
        points.extend(
            get_circle_vertices(
                np.append(center, [np.max(points, axis=0)[2]]),
                0.9 * radius,
                n=int(math.ceil(360.0 / 5)),
            )
        )
        # draw_points(points)
        # wait_if_gui()
        # trimesh.path.simplify

    points_origin = tform_points(invert(origin_pose), points)
    # base_points = project_base_points(points, min_z=min_z, max_z=max_z)
    base_points = project_points(points, min_z=min_z)
    base_origin = tform_points(invert(origin_pose), base_points)
    base_origin = filter_visible(
        base_origin + points_origin, origin_pose, camera_image
    )  # + points_origin

    # TODO: outlier removal and voxel downsampling
    # TODO: getContactPoints/getClosestPoints: contactDistance (negative for penetration)
    # TODO: buffer using grow_polygon
    if use_geom or (sc_network is None):
        merged_origin = base_origin if project_base else points_origin
        obj_mesh = (
            concave_mesh(merged_origin) if concave else mesh_from_points(merged_origin)
        )  # concave_mesh | concave_hull
        if obj_mesh is None:
            return None
    else:  # estimate mesh using DNNs
        obj_mesh = refine_shape(
            sc_network, points_origin, use_points=use_points, min_z=min_z, **kwargs
        )
        if use_image and (
            camera_image is not None
        ):  # TODO: account for the change in frame
            obj_mesh = Mesh(
                filter_visible(
                    obj_mesh.vertices, origin_pose, camera_image, instance=instance
                ),
                faces=None,
            )
            # assert len(obj_mesh.vertices) >= 3
            if len(obj_mesh.vertices) < min_points:
                return None
        if use_hull or (obj_mesh.faces is None):
            # TODO: unify with the above
            merged_origin = obj_mesh.vertices
            if use_points:
                # Could also project obj_mesh.vertices
                merged_origin = np.vstack(
                    [merged_origin, base_origin if project_base else points_origin]
                )
            obj_mesh = (
                concave_mesh(merged_origin)
                if concave
                else mesh_from_points(merged_origin)
            )

        if len(obj_mesh.vertices) >= 3:
            obj_mesh = trim_mesh(obj_mesh)
    # TODO: check min_volume

    # TODO: remove outliers from shape completion
    # import open3d
    # open3d.voxel_down_sample
    # open3d.uniform_down_sample
    # open3d.remove_statistical_outlier
    # open3d.remove_radius_outlier
    # open3d.sample_points_poisson_disk

    # draw_mesh(obj_mesh, color=color)
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
    set_pose(obj_estimate, origin_pose, **kwargs)
    # set_all_color(obj_estimate, color)

    # TODO: return an EstimatedObjects that stores points, visibility, etc
    return obj_estimate


def estimate_surface_mesh(
    labeled_points, surface_pose=None, camera_image=None, **kwargs
):
    # TODO: surface instead of surface_pose
    if surface_pose is None:
        min_z = min(lp.point[2] for lp in labeled_points)
        surface_pose = Pose(Point(z=min_z))
    if camera_image is not None:
        rgb, depth, labeled, camera_pose = camera_image[:4]
        camera_pose = multiply(invert(surface_pose), camera_pose)
        camera_image = CameraImage(rgb, depth, labeled, camera_pose, *camera_image[4:])

    labeled_cluster = tform_labeled_points(invert(surface_pose), labeled_points)
    body = estimate_mesh(
        labeled_cluster, camera_image=camera_image, **kwargs
    )  # assumes the surface is an xy plane
    if body is None:
        return body
    set_pose(body, multiply(surface_pose, get_pose(body, **kwargs)), **kwargs)
    return body
