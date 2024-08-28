import math
import os
import random
from itertools import count

import numpy as np
import open3d
import pybullet as p

import owt.pb_utils as pbu
from owt.utils import TEMP_DIR

DEFAULT_ALPHA = 0.015
VHACD_CNT = count()


def create_vhacd(input_path, output_path=None, client=None, **kwargs):
    client = client or p
    if output_path is None:
        output_path = os.path.join(pbu.TEMP_DIR, "vhacd_{}.obj".format(next(VHACD_CNT)))
    log_path = os.path.join(pbu.TEMP_DIR, "vhacd_log.txt")
    with pbu.HideOutput(enable=True):
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
    return pbu.create_obj(output_path, **kwargs)


def decompose_body(body):
    visdata = pbu.get_visual_data(body)[0]
    vhacd_file_in = visdata.meshAssetFileName.decode()
    color = visdata.rgbaColor
    obj_pose = pbu.get_pose(body)
    pbu.remove_body(body)
    new_body = create_vhacd(vhacd_file_in, color=color)
    pbu.set_pose(new_body, obj_pose)
    return new_body


def obj_file_from_mesh(mesh, under=True):
    """Creates a *.obj mesh string :param mesh: tuple of list of vertices and
    list of faces :return: *.obj mesh string."""
    vertices, faces = mesh
    s = "g Mesh\n"  # TODO: string writer
    for v in vertices:
        assert len(v) == 3
        s += "\nv {}".format(" ".join(map(str, v)))
    for f in faces:
        # assert(len(f) == 3) # Not necessarily true
        f = [i + 1 for i in f]  # Assumes mesh is indexed from zero
        s += "\nf {}".format(" ".join(map(str, f)))
        if under:
            s += "\nf {}".format(" ".join(map(str, reversed(f))))
    return s


mesh_count = count()


def create_mesh(mesh, under=True, **kwargs):
    pbu.ensure_dir(TEMP_DIR)
    path = os.path.join(TEMP_DIR, "mesh{}.obj".format(next(mesh_count)))
    pbu.write(path, obj_file_from_mesh(mesh, under=under))
    return pbu.create_obj(path, **kwargs)


def write(filename, string):
    with open(filename, "w") as f:
        f.write(string)


def create_concave_mesh(mesh, under=False, client=None, **kwargs):
    pbu.ensure_dir(TEMP_DIR)
    num = next(mesh_count)
    path = os.path.join(TEMP_DIR, "mesh{}.obj".format(num))
    write(path, obj_file_from_mesh(mesh, under=under))
    vhacd_path = os.path.join(TEMP_DIR, "vhacd_mesh{}.obj".format(num))
    return create_vhacd(path, output_path=vhacd_path, **kwargs)


def recon_bpa(points, debug=False, **kwargs):
    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(np.asarray(points)))
    if debug:
        open3d.visualization.draw_geometries([pcd])

    pcd.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 4 * avg_dist
    # larger radius -> convex hull
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, open3d.utility.DoubleVector([radius, radius * 2])
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return pbu.Mesh(mesh.vertices, mesh.triangles)


def recon_poisson(points, debug=False, **kwargs):
    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(np.asarray(points)))
    if debug:
        open3d.visualization.draw_geometries([pcd])

    pcd.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )

    # remove some extrapolated points
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    return pbu.Mesh(mesh.vertices, mesh.triangles)


def recon_alpha_shape(points, alpha=DEFAULT_ALPHA, debug=False, **kwargs):
    assert alpha > 0.0
    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(np.asarray(points)))
    if debug:
        open3d.visualization.draw_geometries([pcd])

    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=alpha
    )
    return pbu.Mesh(mesh.vertices, mesh.triangles)


##################################################


def concave_hull(points, use_poisson=False, use_bpa=False, **kwargs):
    if use_poisson:
        return recon_poisson(points, **kwargs)
    if use_bpa:
        return recon_bpa(points, **kwargs)
    return recon_alpha_shape(points, **kwargs)


def downsample_aidan(points, fraction=1.0 / 500):
    points_array = np.asarray(points)
    # There is some sort of bug in open3d so we need to downsample
    ds = max(3, math.ceil(fraction * points_array.shape[0]))
    downsampled_pointcloud = points_array[::ds, :]
    return downsampled_pointcloud


def safe_sample(collection, k=1):
    collection = list(collection)
    if len(collection) <= k:
        return collection
    return random.sample(collection, k)


def concave_mesh(points, alpha=DEFAULT_ALPHA, **kwargs):
    sampled_points = safe_sample(points, k=np.inf)
    mesh = concave_hull(sampled_points, alpha=alpha, **kwargs)
    if mesh is None:
        return pbu.mesh_from_points(points)
    return mesh
