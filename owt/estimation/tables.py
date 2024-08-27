import math
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans

import owt.pb_utils as pbu
from owt.estimation.clustering import cluster_trimesh
from owt.estimation.observation import (aggregate_color, extract_point,
                                        iterate_image, iterate_point_cloud,
                                        save_camera_images)
from owt.estimation.surfaces import (Plane, compute_inliers, create_surface,
                                     plane_from_pose, point_plane_distance,
                                     ransac_estimate_plane)
from owt.simulation.entities import TABLE, UNKNOWN, Label, Table
from owt.simulation.utils import (Z_AXIS, find_closest_color,
                                  get_color_distance, mean_hue, select_indices)

COLORS = {
    "yellow": pbu.YELLOW,
}
# COLORS.update(CHROMATIC_COLORS)

# TODO: infer from the goal statement or explicitly specify in the problem
SURFACE_COLORS = {
    # TABLE: [0.855, 0.733, 0.612], # 0.596
    "red": [0.737, 0.082, 0.227],  # 0.044
    "yellow": [0.953, 0.89, 0.169],  # 0.043
    "green": [
        0.592,
        0.804,
        0.353,
    ],  # [0.631, 0.902, 0.463], # Green seems hard to detect?
    "blue": [0.31, 0.431, 0.804],
}
COLORS.update(SURFACE_COLORS)

##################################################


def extract_table(camera_image, **kwargs):
    points_from_label = defaultdict(list)
    for lp in iterate_point_cloud(camera_image, **kwargs):
        points_from_label[tuple(lp.label)].append(lp)

    table_points = []
    pos_table_points = []
    for (category, instance), points in points_from_label.items():
        pos_table_points.extend(lp for lp in points)
    return table_points, pos_table_points


def relabel_table(
    camera_image,
    plane,
    threshold=3e-2,
    check_all=True,
    ignore_obj=False,
    min_z=0.3,
    **kwargs
):
    # ignore_obj: set to True to relabel all pixels close to table, ignore obj masks.
    # TODO: image-based operations like taking the 2D AABB or OOBB
    start_time = time.time()
    check_pixels = (
        set(iterate_image(camera_image, step_size=4, **kwargs)) if check_all else set()
    )
    table_label = Label(TABLE, TABLE)
    unknown_label = Label(UNKNOWN, UNKNOWN)
    seg = camera_image[2]
    num_relabeled = 0
    for pixel in iterate_image(camera_image, step_size=1, **kwargs):
        category, instance = seg[pixel]
        if category == TABLE:  # Single table instance
            seg[pixel] = table_label
        elif ignore_obj or (category != UNKNOWN) or (pixel in check_pixels):
            labeled_point = extract_point(camera_image, pixel=pixel)
            if labeled_point.point[2] < min_z:
                seg[pixel] = unknown_label
                continue
            if abs(point_plane_distance(plane, labeled_point.point)) <= threshold:
                # seg[pixel] = table_label
                seg[pixel] = unknown_label
                num_relabeled += 1
    print(
        "Relabeled {} points in {:.3f} sec".format(
            num_relabeled, pbu.elapsed_time(start_time)
        )
    )
    return seg


##################################################


def estimate_region(
    labeled_points,
    plane=None,
    threshold=5e-2,
    min_side=8e-2,
    thickness=1e-2,
    category=None,
    **kwargs
):  # Squares are 0.2 x 0.2

    points = list(set([lp.point for lp in labeled_points]))
    # points = safe_sample(points, k=500)
    # draw_points(points)

    if plane is None:
        plane, _ = ransac_estimate_plane(
            points, threshold=threshold, max_error=math.radians(1)
        )
    else:
        distances = [
            point_plane_distance(plane, point, signed=True) for point in points
        ]
        median_distance = max(1e-3, np.median(distances))
        plane = Plane(plane.normal, plane.origin + median_distance * plane.normal)

    if plane is None:
        return None

    print(
        "Angle error: {:.3f} degrees".format(
            math.degrees((pbu.angle_between(plane.normal, Z_AXIS)))
        )
    )
    indices = compute_inliers(plane, points, threshold=threshold)
    inliers = select_indices(points, indices)

    surface = create_surface(
        plane, inliers, max_distance=threshold, min_area=min_side**2, **kwargs
    )

    if surface is None:
        return None

    vertices, pose = surface
    surface_aabb = pbu.aabb_from_points(vertices)
    if any(side < min_side for side in pbu.get_aabb_extent(surface_aabb)[:2]):
        return None
    vertices = pbu.get_aabb_vertices(surface_aabb)

    if thickness == 0:
        mesh_points = vertices
    else:
        mesh_points = vertices + [
            vertex - thickness * np.array(Z_AXIS) for vertex in vertices
        ]

    labeled_points = select_indices(labeled_points, indices)
    color = aggregate_color(labeled_points)
    colors = [lp.color for lp in labeled_points]
    color_name = find_closest_color(color, color_from_name=COLORS, hue_only=False)
    print(
        color_name,
        color,
        mean_hue(colors),
        get_color_distance(color, COLORS[color_name], hue_only=False),
    )
    if category is None:
        category = "{}_region".format(color_name)

    mesh = pbu.mesh_from_points(mesh_points)  # vertices | mesh_points
    body = pbu.create_mesh(mesh, under=False, color=pbu.apply_alpha(color), **kwargs)
    pbu.set_pose(body, pose, **kwargs)
    return Table(
        surface, body, category=category, color=color, points=mesh_points, **kwargs
    )


def cluster_inliers(inliers, max_clusters=2):  # TODO another kind of clustering?
    if not inliers:
        return []
    # from scipy.cluster.vq import kmeans, kmeans2
    # https://scikit-learn.org/stable/modules/clustering.html
    kmeans = KMeans(
        n_clusters=max_clusters,  # init='k-means++',
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    )
    kmeans.fit(inliers)  # TODO: different color space
    indices_from_label = {}
    for index, label in enumerate(kmeans.labels_):
        indices_from_label.setdefault(label, []).append(index)
    labels = sorted(
        indices_from_label, key=lambda l: len(indices_from_label[l]), reverse=True
    )
    return [indices_from_label[label] for label in labels]


def cluster_colors(colors, max_clusters=len(COLORS)):
    if not colors:
        return []
    # from scipy.cluster.vq import kmeans, kmeans2
    # https://scikit-learn.org/stable/modules/clustering.html
    kmeans = KMeans(
        n_clusters=max_clusters,  # init='k-means++',
        init=np.array(list(map(pbu.remove_alpha, COLORS.values()))),
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    )
    kmeans.fit(colors)  # TODO: different color space
    # TODO: repeatedly cluster and remove nearby colors

    indices_from_label = {}
    for index, label in enumerate(kmeans.labels_):
        indices_from_label.setdefault(label, []).append(index)
    labels = sorted(
        indices_from_label, key=lambda l: len(indices_from_label[l]), reverse=True
    )

    print(
        "Colors distances:",
        {
            (c1, c2): get_color_distance(
                kmeans.cluster_centers_[c1], kmeans.cluster_centers_[c2], hue_only=False
            )
            for c1, c2 in combinations(labels, r=2)
        },
    )

    return [indices_from_label[label] for label in labels]


def cluster_known(colors, max_distance=0.25, hue_only=False):
    indices_from_label = {}
    for index, color in enumerate(colors):
        color_name = find_closest_color(
            color, color_from_name=COLORS, hue_only=hue_only
        )
        if (
            get_color_distance(color, COLORS[color_name], hue_only=hue_only)
            <= max_distance
        ):
            indices_from_label.setdefault(color_name, []).append(index)
    labels = sorted(
        indices_from_label, key=lambda l: len(indices_from_label[l]), reverse=True
    )
    return [indices_from_label[label] for label in labels]


def estimate_regions(
    camera_image,
    min_points=100,
    plane_threshold=5e-2,
    color_threshold=0.3,
    min_z=0.3,
    save_relabled=False,
    **kwargs
):

    _, pos_table_points = extract_table(camera_image, **kwargs)
    extracted_pos_table = list(set([lp.point for lp in pos_table_points]))
    table_plane, inliers = ransac_estimate_plane(
        extracted_pos_table, threshold=plane_threshold, max_error=math.radians(1)
    )
    # table = estimate_region(table_points, **kwargs)
    # table_plane = plane_from_pose(table.surface.pose)
    # relabel_table(camera_image, table_plane, threshold=2*plane_threshold, min_z=min_z)

    if save_relabled:
        save_camera_images(camera_image, prefix="relabeled_")
    # table_points = [table_points[index] for index in inliers] # TODO: use inliers

    pos_table_points = [
        pos_table_points[index]
        for index in compute_inliers(
            table_plane,
            [lp.point for lp in pos_table_points],
            threshold=plane_threshold,
        )
    ]  # TODO: use inliers

    table = estimate_region(
        pos_table_points, threshold=plane_threshold, category="table", **kwargs
    )
    table_plane = plane_from_pose(table.surface.pose)
    table_color = table.color
    if not COLORS:
        return [table]

    # region_points = table_points
    region_points = [
        lp
        for lp in pos_table_points
        if get_color_distance(lp.color, table_color, hue_only=False) >= color_threshold
    ]
    region_colors = [pbu.remove_alpha(lp.color) for lp in region_points]

    # TODO: remove all points close to this
    clustered_indices = cluster_colors(region_colors, max_clusters=len(COLORS))

    # clustered_indices = cluster_known(region_colors)
    regions = []
    for i, indices in enumerate(clustered_indices):
        # color = kmeans.cluster_centers_[label]
        label_points = [region_points[index] for index in indices]
        color = np.mean([lp.color for lp in label_points], axis=0)
        clusters = cluster_trimesh(
            label_points, noise_only=False, radius=5e-2, use_2d=False
        )
        for cluster in clusters[:1]:
            cluster_points = [label_points[index] for index in cluster]
            if len(cluster_points) < min_points:
                continue
            print(
                "Label: {} | Size: {} | Color: {} | Distance: {:.3f} |".format(
                    i,
                    len(cluster_points),
                    color.round(3).tolist(),
                    get_color_distance(table_color, color, hue_only=False),
                ),
                list(map(len, clusters)),
            )
            # region_plane = None
            region_plane = table_plane
            # region_plane = Plane(table_plane.normal, table_plane.origin + 1e-3*table_plane.normal)
            region = estimate_region(
                cluster_points, plane=region_plane, threshold=plane_threshold, **kwargs
            )
            if region is not None:
                regions.append(region)
                # wait_if_gui()
    return [table] + regions


def estimate_surfaces(belief, camera_image, **kwargs):
    surfaces = estimate_regions(camera_image, **kwargs)
    assert surfaces  # TODO: while true
    belief.known_surfaces = surfaces

    regions = surfaces[1:]
    for surface in regions:
        color = pbu.get_color(surface, **kwargs)
        color_name = find_closest_color(color, color_from_name=SURFACE_COLORS)
        # print(surface, np.array(color), color_name)
        surface.properties.extend(
            [
                ("Region",),
                ("Color", color_name),
            ]
        )
    return surfaces
