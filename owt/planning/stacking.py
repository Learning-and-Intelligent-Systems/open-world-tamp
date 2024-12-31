import traceback
from collections import defaultdict, deque
from itertools import combinations

import numpy as np
from trimesh.intersections import mesh_multiplane, mesh_plane
from trimesh.path.exchange.misc import lines_to_path, polygon_to_path
from trimesh.path.path import Path2D
from trimesh.path.polygons import plot, projected

from owt.estimation.surfaces import create_surface, z_plane


def get_connected_components(vertices, edges):
    undirected_edges = defaultdict(set)
    for v1, v2 in edges:
        undirected_edges[v1].add(v2)
        undirected_edges[v2].add(v1)
    clusters = []
    processed = set()
    for v0 in vertices:
        if v0 in processed:
            continue
        processed.add(v0)
        cluster = {v0}
        queue = deque([v0])
        while queue:
            v1 = queue.popleft()
            for v2 in undirected_edges[v1] - processed:
                processed.add(v2)
                cluster.add(v2)
                queue.append(v2)
        if cluster:  # preserves order
            clusters.append(frozenset(cluster))
    return clusters


def cluster_identical(points, tolerance=1e-6):
    indices = list(range(len(points)))
    pairs = set()
    for index1, index2 in combinations(indices, r=2):
        if np.allclose(points[index1], points[index2], rtol=0.0, atol=tolerance):
            pairs.add((index1, index2))
    components = get_connected_components(indices, pairs)
    return components


def manual_slice_mesh(mesh, plane=z_plane()):
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
    plane_normal, plane_origin = plane
    try:
        polygon_shapely = projected(
            mesh,
            plane_normal,
            origin=plane_origin,
            pad=1e-05,
            tol_dot=0.01,
            max_regions=1000,
        )
    except ValueError:
        traceback.print_exc()
        return []
    plot(polygon_shapely)
    polygon = Path2D(**polygon_to_path(polygon_shapely))
    return polygon


def slice_mesh(mesh, plane=z_plane()):
    plane_normal, plane_origin = plane
    lines = mesh_plane(
        mesh, plane_normal, plane_origin, return_faces=False, cached_dots=None
    )  # (n, 2, 3)
    if lines.shape[0] == 0:
        return []
    points = np.vstack(lines)
    surface = create_surface(plane, points, origin_type="centroid")
    if surface is None:
        return []
    return [surface]
