import time

import numpy as np
from pybullet_tools.utils import (
    BASE_LINK,
    GREEN,
    LockRenderer,
    Mesh,
    convex_hull,
    draw_mesh,
    draw_point,
    draw_pose,
    elapsed_time,
    get_com_pose,
    get_joint_inertial_pose,
    get_link_inertial_pose,
    get_link_pose,
    get_mesh_data,
    get_mesh_geometry,
    get_mesh_normal,
    get_model_info,
    mesh_count,
    remove_handles,
    vertices_from_link,
    vertices_from_rigid,
    wait_if_gui,
)

from open_world.estimation.observation import draw_points
from open_world.estimation.surfaces import Plane, point_plane_distance


def test_trimish():
    # https://trimsh.org/trimesh.html
    # from trimesh import Trimesh
    # Trimesh.contains
    # Trimesh.face_adjacency_convex
    # Trimesh.face_adjacency_projections
    # Trimesh.simplify_quadratic_decimation
    # Trimesh.voxelized
    # from trimesh.voxel import VoxelGrid
    # from trimesh.points import plane_fit
    # from trimesh.boolean import intersection, union, difference
    # from trimesh.caching import cache_decorator
    # from trimesh.collision import CollisionManager
    # from trimesh.curvature import line_ball_intersection
    # from trimesh.decomposition import convex_decomposition
    # from trimesh.exchange import dae, obj, off, ply, stl, urdf
    # from trimesh.nsphere import minimum_nsphere, fit_nsphere
    # from trimesh.path.packing import paths, polygons, rectangles
    # from trimesh.path.polygons import projected, plot_polygon
    # from trimesh.poses import compute_stable_poses
    # from trimesh.proximity import ProximityQuery, closest_point, signed_distance
    # ProximityQuery.on_surface
    # ProximityQuery.signed_distance
    # from trimesh.permutate import transform
    # from trimesh.ray.ray_pyembree import RayMeshIntersector
    # from trimesh.ray.ray_triangle import RayMeshIntersector, ray_bounds
    # RayMeshIntersector.contains_points
    # RayMeshIntersector.intersects_any
    # from trimesh.ray.ray_util import contains_points
    # from trimesh.registration import icp, mesh_other
    # from trimesh.repair import fill_holes
    # from trimesh.sample import volume_mesh, volume_rectangular, sample_surface_even, sample_surface
    pass


def test_open3d():
    import open3d

    open3d.geometry.PointCloud
    open3d.geometry.TriangleMesh.create_cylinder

    # from open3d.visualization import draw_geometries
    pcd = open3d.io.read_point_cloud("../../test_data/fragment.ply")
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=0.03
    )

    mesh.compute_vertex_normals()
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=5)
    pcd = mesh.sample_points_uniformly(number_of_points=500)
    mesh.cluster_connected_triangles()

    reg_p2p = open3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )


def estimate_pcl_plane(points):
    # https://numpy.org/doc/stable/reference/routines.linalg.html#matrix-eigenvalues
    # https://github.com/strawlab/python-pcl
    # https://github.com/daavoo/pyntcloud
    # https://github.com/davidcaron/pclpy
    # https://github.com/ajhynes7/scikit-spatial
    # https://github.com/falcondai/py-ransac
    # https://github.com/leomariga/pyRANSAC-3D
    # https://trimsh.org/trimesh.points.html?highlight=plane#trimesh.points.plane_fit
    # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html?highlight=segment_plane#Plane-segmentation
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
    # https://scipy-cookbook.readthedocs.io/items/RANSAC.html
    # https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    # https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html?highlight=plane
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_perception/blob/master/perception_tools/visual_octomap.cpp#L405
    # https://github.com/fwilliams/point-cloud-utils
    # https://github.com/mmolero/pcloudpy
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
    import pcl

    cloud = pcl.PointCloud()
    cloud.from_array(points)
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_normal_distance_weight(0.05)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.005)
    inliers, model = seg.segment()
    # TODO: include the base plane
    raise NotImplementedError()


def test_properties(body):
    # loadURDF: basePosition
    # create the base of the object at the specified position in world space coordinates [X,Y,Z].
    # Note that this position is of the URDF link position.
    # If the inertial frame is non-zero, this is different from the center of mass position.
    # Use resetBasePositionAndOrientation to set the center of mass location/orientation.
    # p.URDF_MERGE_FIXED_LINKS
    # p.URDF_USE_INERTIA_FROM_FILE # by default, Bullet recomputed the inertia tensor based on mass and volume of the collision shape.
    # p.URDF_USE_SELF_COLLISION # by default, Bullet disables self-collision
    # p.URDF_MAINTAIN_LINK_ORDER

    # createMultiBody: basePosition
    # Cartesian world position of the base

    link = BASE_LINK
    pose = get_joint_inertial_pose(body, link)
    pose = get_com_pose(body, link)
    pose = get_link_inertial_pose(body, link)
    pose = get_link_pose(body, link)

    draw_pose(pose)

    info = get_model_info(body)
    assert info is not None
    info.path

    vertices = vertices_from_rigid(body)  # get_collision_data
    vertices = vertices_from_link(body)  # RuntimeError: unknown_file
    # vertices = hull.vertices
    with LockRenderer():
        for vertex in vertices:
            draw_point(vertex)

    wait_if_gui()


def hull_ransac(points, min_points=10, threshold=5e-3, draw=False):
    # TODO: plane estimation instead of RANSAC here (in case bad orientations)
    start_time = time.time()
    hull = Mesh(*map(np.array, convex_hull(points)))
    centroid = np.average(hull.vertices, axis=0)
    if draw:
        with LockRenderer():
            draw_mesh(hull)
    # equations = []
    planes = []
    while len(points) >= min_points:
        best_plane, best_indices = None, []
        for i, face in enumerate(hull.faces):
            v1, v2, v3 = hull.vertices[face]
            # normal = get_normal(v1, v2, v3)
            normal = get_mesh_normal(hull.vertices[face], centroid)
            plane = Plane(normal, v1)
            # equation = np.array(equation_from_plane(plane))
            # if equation[3] > 0: # TODO: cluster by plane equation
            #     equation *= 1
            # TODO: shouldn't need abs
            indices = [
                index
                for index, point in enumerate(points)
                if abs(point_plane_distance(plane, point)) <= threshold
            ]  # TODO: cache
            if len(indices) > len(best_indices):
                best_plane, best_indices = plane, indices
            # TODO: weigh by area
            # TODO: density of the space covered
        if len(best_indices) < min_points:
            break
        print(
            "{} | {} | {} | {:.3f}".format(
                len(planes), best_plane, len(best_indices), elapsed_time(start_time)
            )
        )
        if draw:
            handles = draw_points(
                [points[index] for index in best_indices], color=GREEN
            )
            wait_if_gui()
            remove_handles(handles)
        planes.append(best_plane)
        points = [
            point for index, point in enumerate(points) if index not in best_indices
        ]
    return planes
