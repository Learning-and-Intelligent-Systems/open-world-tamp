import itertools
from collections import Counter
from itertools import product

import numpy as np
import open3d
import trimesh
from sklearn.cluster import DBSCAN

import owt.pb_utils as pbu
from owt.estimation.observation import (LabeledPoint, aggregate_color,
                                        draw_points)
from owt.simulation.entities import UNKNOWN, Label
from owt.simulation.utils import select_indices


def has_open3d():
    try:
        pass
    except OSError:
        return False
    return True


def relabel_nearby(labeled_points, max_distance=3e-2):
    indices = list(range(len(labeled_points)))
    edges = set()
    for index1, index2 in product(indices, repeat=2):
        if (
            pbu.get_distance(labeled_points[index1].point, labeled_points[index2].point)
            < max_distance
        ):
            edges.add((index1, index2))

    neighbors_from_vertex = {}
    for index1, index2 in edges:
        neighbors_from_vertex.setdefault(index1, []).append(index2)

    for index1 in neighbors_from_vertex:
        point, color, label = labeled_points[index1]
        category, instance = label
        if category != UNKNOWN:
            continue
        candidates = [
            index
            for index in neighbors_from_vertex[index1]
            if labeled_points[index].label[0] != UNKNOWN
        ]
        if not candidates:
            continue
        closest_index = min(
            candidates, key=lambda i: pbu.get_distance(point, labeled_points[i].point)
        )
        labeled_points[index1] = LabeledPoint(
            point, color, labeled_points[closest_index].label
        )
    return labeled_points


def cluster_unassigned(labeled_points, groups, **kwargs):
    assigned_indices = set(itertools.chain(groups))
    indices = list(range(len(labeled_points)))
    unknown_indices = [index for index in indices if index not in assigned_indices]
    if not unknown_indices:
        return groups
    original_from_unknown = dict(enumerate(unknown_indices))
    new_groups = cluster_trimesh(
        [labeled_points[index] for index in unknown_indices], **kwargs
    )
    return groups + [
        [original_from_unknown[index] for index in group] for group in new_groups
    ]


def sort_clusters(groups):
    return sorted(groups, key=len, reverse=True)


def remove_outlier_group(points, sub_groups_idx, dist_threshold=0.2):
    if len(sub_groups_idx) <= 1:
        return sub_groups_idx

    points = np.asarray(points)
    cluster_mean = [
        np.take(points, sub_group_idx, 0).mean(0) for sub_group_idx in sub_groups_idx
    ]
    largest_cluster = np.argmax(
        [len(sub_group_idx) for sub_group_idx in sub_groups_idx]
    )
    sub_groups_idx = [
        sub_group_idx
        for i, sub_group_idx in enumerate(sub_groups_idx)
        if np.linalg.norm(cluster_mean[i] - cluster_mean[largest_cluster])
        <= dist_threshold
    ]
    return sub_groups_idx


def remove_outliers_noise_only(
    group, points, sub_groups_idx, noise_only=True, **kwargs
):
    if noise_only:
        sub_groups_idx = remove_outlier_group(points, sub_groups_idx, **kwargs)
    sub_groups = [
        np.take(group, sub_group_idx).tolist() for sub_group_idx in sub_groups_idx
    ]
    if noise_only:
        sub_groups = [sum(sub_groups, [])]  # merge subgroups into one
    return sub_groups


##################################################

DEFAULT_RADIUS = 5e-2  # 3e-2 | 5e-2


def cluster_trimesh(
    labeled_points, groups=None, radius=DEFAULT_RADIUS, use_2d=True, **kwargs
):
    if groups is None:
        groups = [list(np.arange(len(labeled_points)))]
    new_groups = []
    for group in groups:
        # TODO: scale to represent different dimension weights
        points = [
            labeled_points[labeled_point_idx].point for labeled_point_idx in group
        ]
        if use_2d:
            points = [point[:2] for point in points]  # TODO: project instead
        sub_groups_idx = trimesh.grouping.clusters(points, radius)
        new_groups.extend(
            remove_outliers_noise_only(group, points, sub_groups_idx, **kwargs)
        )

    return sort_clusters(new_groups)


def cluster_sklearn(
    labeled_points, groups=None, radius=DEFAULT_RADIUS, min_points=1, **kwargs
):
    if groups is None:
        groups = [list(np.arange(len(labeled_points)))]
    new_groups = []
    for group in groups:
        points = np.vstack(
            [labeled_points[labeled_point_idx].point for labeled_point_idx in group]
        )
        indices = (
            DBSCAN(eps=radius, min_samples=min_points).fit(points).labels_
        )  # the label -1 indicates noise
        if len(indices) == 0:
            continue
        num_clusters = max(indices) + 1
        sub_groups_idx = [[] for _ in range(num_clusters)]
        for index, sub_group in enumerate(indices):
            if sub_group != -1:
                sub_groups_idx[sub_group].append(index)
        new_groups.extend(
            remove_outliers_noise_only(group, points, sub_groups_idx, **kwargs)
        )
    return sort_clusters(new_groups)


def cluster_open3d(
    labeled_points, groups=None, radius=DEFAULT_RADIUS, min_points=1, **kwargs
):
    if groups is None:
        groups = [list(np.arange(len(labeled_points)))]
    new_groups = []
    for group in groups:
        points = [
            labeled_points[labeled_point_idx].point for labeled_point_idx in group
        ]
        cloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
        indices = cloud.cluster_dbscan(
            eps=radius, min_points=min_points
        )  # the label -1 indicates noise
        if len(indices) == 0:
            continue
        num_clusters = max(indices) + 1
        sub_groups_idx = [[] for _ in range(num_clusters)]
        for index, sub_group in enumerate(indices):
            if sub_group != -1:
                sub_groups_idx[sub_group].append(index)
        new_groups.extend(
            remove_outliers_noise_only(group, points, sub_groups_idx, **kwargs)
        )
    return sort_clusters(new_groups)


def cluster_segmented(labeled_points):
    labeled_group = {}
    for index, label_point in enumerate(labeled_points):
        category, instance = label_point.label
        labeled_group.setdefault(instance, []).append(index)
    return sort_clusters(labeled_group.values())


def dump_groups(labeled_points, groups):
    print("Clusters ({}):".format(len(groups)))
    for i, group in enumerate(groups):
        # TODO: integrate with relabel_clusters
        group_points = select_indices(labeled_points, group)
        [(cls_label, _)] = Counter(lp.label[0] for lp in group_points).most_common(1)
        [(ins_label, _)] = Counter(lp.label[1] for lp in group_points).most_common(1)
        print(
            "Cluster: {} | Size: {} | Category: {} | Instance: {}".format(
                i, len(group), cls_label, ins_label
            )
        )


##################################################


def cluster_points(
    labeled_points,
    min_points=1,
    use_instance_label=True,
    use_open3d=False,
    use_trimesh=True,
    use_sklearn=False,
    draw=False,
    noise_only=False,
    **kwargs
):
    """
    use_instance_label: cluster in label space
    noise_only:         use geometric clustering for noise filtering only(do not decompose the object into subgroups)
    dist_threshold:     largest distance allowed between subgroups
    """

    assert (not noise_only) or use_instance_label

    if len(labeled_points) < min_points:
        return []

    clusters = [list(np.arange(len(labeled_points)))]
    if use_instance_label:
        clusters = cluster_segmented(labeled_points)
        clusters = cluster_unassigned(labeled_points, clusters, radius=3e-2)
        dump_groups(labeled_points, clusters)
    if use_open3d:
        clusters = cluster_open3d(
            labeled_points,
            clusters,
            min_points=min_points,
            noise_only=noise_only,
            **kwargs
        )
    if use_trimesh:
        clusters = cluster_trimesh(
            labeled_points, clusters, noise_only=noise_only, **kwargs
        )
    if use_sklearn:
        clusters = cluster_sklearn(
            labeled_points, clusters, noise_only=noise_only, **kwargs
        )

    clusters = sort_clusters(
        tuple(cluster) for cluster in clusters if len(cluster) >= min_points
    )

    dump_groups(labeled_points, clusters)

    if draw:
        handles = []
        for group in clusters:
            group_points = select_indices(labeled_points, group)
            handles.extend(
                draw_points(
                    [lp.point for lp in group_points],
                    color=aggregate_color(group_points),
                )
            )
        pbu.wait_if_gui()
    return clusters


def relabel_clusters(labeled_points, **kwargs):
    # from statistics import mode
    for cluster in cluster_points(labeled_points, **kwargs):
        points = select_indices(labeled_points, cluster)
        instances = Counter(point.label[1] for point in points)
        [(instance_label, _)] = instances.most_common(1)

        categories = Counter(point.label[0] for point in points)
        [(category_label, _)] = categories.most_common(1)

        new_label = Label(category_label, instance_label)
        relabeled_points = [
            LabeledPoint(point.point, point.color, new_label) for point in points
        ]
        yield relabeled_points
