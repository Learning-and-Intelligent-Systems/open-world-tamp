import traceback
from itertools import combinations

import numpy as np
from pybullet_tools.utils import get_connected_components, pose_from_tform

from open_world.estimation.surfaces import create_surface, z_plane


def cluster_identical(points, tolerance=1e-6):
    # from trimesh.path.repair import fill_gaps
    # TODO: instead fill_gaps just adds edges for nearby vertices
    indices = list(range(len(points)))
    pairs = set()
    for index1, index2 in combinations(indices, r=2):
        if np.allclose(points[index1], points[index2], rtol=0.0, atol=tolerance):
            pairs.add((index1, index2))
    components = get_connected_components(indices, pairs)
    return components


def manual_slice_mesh(mesh, plane=z_plane()):
    from trimesh.intersections import mesh_multiplane
    from trimesh.path.exchange.misc import lines_to_path
    from trimesh.path.path import Path2D

    # segments = mesh_plane(mesh, plane_normal, plane_origin,
    #                      return_faces=False, cached_dots=None) # (n, 2, 3)
    # return [(segments[i, 0, :], segments[i, 1, :]) for i in range(segments.shape[0])]

    plane_normal, plane_origin = plane
    [lines], [tform], [faces] = mesh_multiplane(
        mesh, plane_origin, plane_normal, heights=[0.0]
    )
    points = np.vstack(lines)

    vertices = []
    vertex_from_index = {}
    for component in cluster_identical(points):
        component = sorted(component)
        vertex_from_index.update({index: len(vertices) for index in component})
        vertices.append(points[component[0]])
    edges = [
        tuple(vertex_from_index[2 * i + k] for k in range(2)) for i in range(len(lines))
    ]

    edges = np.array(edges)
    vertices = np.array(vertices)
    path = Path2D(**lines_to_path(lines))
    path.show()

    return vertices, edges


def project_mesh(mesh, plane=z_plane()):
    from trimesh.path.exchange.misc import polygon_to_path
    from trimesh.path.path import Path2D
    from trimesh.path.polygons import plot_polygon, projected

    plane_normal, plane_origin = plane
    # mesh = slice_mesh_plane(mesh, -plane_normal, plane_origin)
    try:
        polygon_shapely = projected(
            mesh,
            plane_normal,
            origin=plane_origin,
            pad=1e-05,
            tol_dot=0.01,
            max_regions=1000,
        )
        # polygon_shapely = projected(mesh, *z_plane(z=0.), pad=1e-05, tol_dot=0.01, max_regions=1000)
    except ValueError:
        traceback.print_exc()
        return []
    plot_polygon(polygon_shapely)
    # print(polygon_shapely.__dict__, dir(polygon_shapely))
    polygon = Path2D(
        **polygon_to_path(polygon_shapely)
    )  # TODO: AttributeError: 'MultiPolygon' object has no attribute 'exterior'
    return polygon


def shapely_slice_mesh(mesh, plane=z_plane()):
    from trimesh.intersections import mesh_multiplane
    from trimesh.path.exchange.load import load_path
    from trimesh.path.polygons import plot_polygon

    plane_normal, plane_origin = plane
    [lines], [tform], [faces] = mesh_multiplane(
        mesh, plane_origin, plane_normal, heights=[0.0]
    )
    pose = pose_from_tform(tform)

    # Could also use create_surface
    # https://trimsh.org/trimesh.path.path.html
    # https://trimsh.org/trimesh.path.entities.html
    path = load_path(lines)
    # path.show()

    # https://shapely.readthedocs.io/en/stable/manual.html#polygons
    attributes = [
        "area",
        "bounds",
        "centroid",
        "is_closed",
        "is_empty",
        "is_ring",
        "is_simple",
        "is_valid",
    ]  # 'boundary', 'exterior',
    for polygon in path.polygons_full:  # polygons_closed | polygons_full
        polygon = polygon.convex_hull
        plot_polygon(polygon)
    quit()

    return []


def slice_mesh(mesh, plane=z_plane()):
    # TODO: could instead use slice_mesh_plane and compute the surface area of the volume
    from trimesh.intersections import mesh_plane

    plane_normal, plane_origin = plane
    lines = mesh_plane(
        mesh, plane_normal, plane_origin, return_faces=False, cached_dots=None
    )  # (n, 2, 3)
    if lines.shape[0] == 0:
        return []
    points = np.vstack(lines)
    # TODO: handle shapes with hollow interiors like cups
    surface = create_surface(plane, points, origin_type="centroid")
    if surface is None:
        return []
    return [surface]
