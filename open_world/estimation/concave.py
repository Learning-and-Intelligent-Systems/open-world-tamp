import math
import os
import time
from itertools import combinations, count

import numpy as np
from pybullet_tools.utils import (
    INF,
    TEMP_DIR,
    HideOutput,
    Mesh,
    create_obj,
    elapsed_time,
    ensure_dir,
    get_connected_components,
    get_pose,
    get_visual_data,
    mesh_count,
    mesh_from_points,
    obj_file_from_mesh,
    orient_face,
    remove_body,
    safe_sample,
    set_pose,
    write,
)

from open_world.estimation.clustering import has_open3d
from open_world.simulation.lis import USING_ROS

DEFAULT_ALPHA = 0.025 if USING_ROS else 0.015  # 0.015 | 0.05 | 0.1 | 1.
VHACD_CNT = count()


def create_vhacd(input_path, output_path=None, client=None, **kwargs):
    import pybullet as p

    client = client or p
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/examples/vhacd.py
    if output_path is None:
        output_path = os.path.join(TEMP_DIR, "vhacd_{}.obj".format(next(VHACD_CNT)))
    log_path = os.path.join(TEMP_DIR, "vhacd_log.txt")
    with HideOutput(enable=True):
        client.vhacd(
            input_path,
            output_path,
            log_path,
            concavity=0.0025,  # Maximum allowed concavity (default=0.0025, range=0.0-1.0)
            alpha=0.04,  # Controls the bias toward clipping along symmetry planes (default=0.05, range=0.0-1.0)
            beta=0.05,  # Controls the bias toward clipping along revolution axes (default=0.05, range=0.0-1.0)
            gamma=0.00125,  # Controls the maximum allowed concavity during the merge stage (default=0.00125, range=0.0-1.0)
            minVolumePerCH=0.0001,  # Controls the adaptive sampling of the generated convex-hulls (default=0.0001, range=0.0-0.01)
            resolution=100000,  # Maximum number of voxels generated during the voxelization stage (default=100,000, range=10,000-16,000,000)
            maxNumVerticesPerCH=64,  # Controls the maximum number of triangles per convex-hull (default=64, range=4-1024)
            depth=20,  # Maximum number of clipping stages. During each split stage, parts with a concavity higher than the user defined threshold are clipped according the best clipping plane (default=20, range=1-32)
            planeDownsampling=4,  # Controls the granularity of the search for the \"best\" clipping plane (default=4, range=1-16)
            convexhullDownsampling=4,  # Controls the precision of the convex-hull generation process during the clipping plane selection stage (default=4, range=1-16)
            pca=0,  # Enable/disable normalizing the mesh before applying the convex decomposition (default=0, range={0,1})
            mode=0,  # 0: voxel-based approximate convex decomposition, 1: tetrahedron-based approximate convex decomposition (default=0,range={0,1})
            convexhullApproximation=1,  # Enable/disable approximation when computing convex-hulls (default=1, range={0,1})
        )
    return create_obj(output_path, **kwargs)


def decompose_body(body):
    visdata = get_visual_data(body)[0]
    vhacd_file_in = visdata.meshAssetFileName.decode()
    color = visdata.rgbaColor
    obj_pose = get_pose(body)
    remove_body(body)
    new_body = create_vhacd(vhacd_file_in, color=color)
    set_pose(new_body, obj_pose)
    return new_body


def create_concave_mesh(mesh, under=False, client=None, **kwargs):
    ensure_dir(TEMP_DIR)
    num = next(mesh_count)
    path = os.path.join(TEMP_DIR, "mesh{}.obj".format(num))
    write(path, obj_file_from_mesh(mesh, under=under))
    vhacd_path = os.path.join(TEMP_DIR, "vhacd_mesh{}.obj".format(num))
    return create_vhacd(path, output_path=vhacd_path, **kwargs)
    # safe_remove(path) # TODO: removing might delete mesh?


##################################################


def query_concave_server(points, alpha=DEFAULT_ALPHA, **kwargs):
    import rospy
    from open_world_server.srv import Hull

    from open_world.real_world.ros_utils import (
        convert_ros_position,
        create_cloud_msg,
        parse_triangle_msg,
    )

    service_name = "/server/hull"
    print("Waiting for service:", service_name)
    rospy.wait_for_service(service_name)
    service = rospy.ServiceProxy(service_name, Hull)

    print(
        "Service: {} | Points: {} | Alpha: {}".format(service_name, len(points), alpha)
    )
    cloud_msg = create_cloud_msg(points)
    # try:
    response = service(cloud_msg, alpha)
    # except rospy.ServiceException as e:
    # print("Service call failed: {}".format(e))
    service.close()

    hulls = response.hulls
    if not hulls:
        return None
    hull = hulls[0]
    vertices = list(map(convert_ros_position, hull.vertices))
    faces = list(map(parse_triangle_msg, hull.triangles))
    return Mesh(vertices, faces)


def pcl_concave_hull(points, alpha=DEFAULT_ALPHA):
    # https://pointclouds.org/documentation/classpcl_1_1_concave_hull.html
    # https://github.com/strawlab/python-pcl/blob/1d83d2d7ce9ce2c22ff5855249459bfc22025000/tests/test_surface.py
    # https://github.com/strawlab/python-pcl/blob/1d83d2d7ce9ce2c22ff5855249459bfc22025000/examples/official/Surface/concave_hull_2d.py#L89
    # https://github.com/strawlab/python-pcl/commits/15bd42a0d5ae9e418b3af05d796ab9d43bad3aa9/tests/test_surface.py
    import pcl

    surf = pcl.ConcaveHull()
    # surf = cloud.make_ConcaveHull()
    surf.set_Alpha(alpha=alpha)
    hull = surf.reconstruct()
    print(hull)
    raise NotImplementedError()


##################################################


def recon_bpa(points, debug=False, **kwargs):
    # https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
    import open3d

    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(np.asarray(points)))
    if debug:
        open3d.visualization.draw_geometries([pcd])

    # Estimate normal + ball-pivoting surface recon
    # pcd = pcd.uniform_down_sample(every_k_points=5)
    pcd.normals = open3d.utility.Vector3dVector(
        np.zeros((1, 3))
    )  # invalidate existing normals
    pcd.estimate_normals()

    # can be very time consuming for large objects. either downsample the pcd or skip this step
    # pcd.orient_normals_consistent_tangent_plane(100)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 4 * avg_dist
    # larger radius -> convex hull
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, open3d.utility.DoubleVector([radius, radius * 2])
    )
    # mesh = mesh.simplify_quadric_decimation(100000)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return Mesh(mesh.vertices, mesh.triangles)


def recon_poisson(points, debug=False, **kwargs):
    # https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
    import open3d

    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(np.asarray(points)))
    if debug:
        open3d.visualization.draw_geometries([pcd])

    # Estimate normal + Poisson surface recon
    # pcd = pcd.uniform_down_sample(every_k_points=5)
    pcd.normals = open3d.utility.Vector3dVector(
        np.zeros((1, 3))
    )  # invalidate existing normals
    pcd.estimate_normals()

    # can be very time consuming for large objects. either downsample the pcd or skip this step
    # pcd.orient_normals_consistent_tangent_plane(100)

    mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )

    # remove some extrapolated points
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    return Mesh(mesh.vertices, mesh.triangles)


def recon_alpha_shape(points, alpha=DEFAULT_ALPHA, debug=False, **kwargs):
    import open3d

    assert alpha > 0.0
    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(np.asarray(points)))
    if debug:
        open3d.visualization.draw_geometries([pcd])
    # NOTE: RuntimeError: [Open3D ERROR] [CreateFromPointCloudAlphaShape] invalid tetra in TetraMesh - Upgrade open3d to latest version(>=0.13.0)
    # https://github.com/intel-isl/Open3D/blob/master/cpp/open3d/geometry/SurfaceReconstructionAlphaShape.cpp#L117

    # Parameter to control the shape. A very big value will give a shape close to the convex hull.
    # http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
    # https://en.wikipedia.org/wiki/Alpha_shape

    # tetra_mesh, pt_map = open3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # tetra_mesh = tetra_mesh.remove_degenerate_tetras()
    # mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)

    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=alpha
    )
    return Mesh(mesh.vertices, mesh.triangles)


##################################################


def concave_hull(points, use_poisson=False, use_bpa=False, debug=False, **kwargs):
    # from trimesh.decomposition import convex_decomposition
    # from pybullet import vhacd
    # TODO: p.vhacd(...) # Volumetric Hierarchical Approximate Decomposition (V-HACD)
    if use_poisson:
        # slow (~45 sec per mesh)
        return recon_poisson(points, **kwargs)
    if use_bpa:
        return recon_bpa(points, **kwargs)
    if not has_open3d() and USING_ROS:
        # return pcl_concave_hull(points, alpha=alpha)
        return query_concave_server(points, **kwargs)
    return recon_alpha_shape(points, **kwargs)


def downsample_aidan(points, fraction=1.0 / 500):
    points_array = np.asarray(points)
    # There is some sort of bug in open3d so we need to downsample
    ds = max(3, math.ceil(fraction * points_array.shape[0]))
    downsampled_pointcloud = points_array[::ds, :]
    return downsampled_pointcloud


def concave_mesh(points, alpha=DEFAULT_ALPHA, **kwargs):
    sampled_points = safe_sample(points, k=INF)
    mesh = concave_hull(sampled_points, alpha=alpha, **kwargs)
    if mesh is None:
        return mesh_from_points(points)
    return mesh