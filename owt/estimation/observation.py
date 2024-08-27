import os
from collections import OrderedDict, namedtuple

import numpy as np

import owt.pb_utils as pbu
from owt.simulation.entities import TABLE, UNKNOWN, Object
from owt.voxels import MAX_PIXEL_VALUE

LabeledPoint = namedtuple("LabeledPoint", ["point", "color", "label"])  # TODO: pixel
BACKGROUND = -1


def aggregate_color(labeled_points):
    colors = [point.color for point in labeled_points]
    return np.median(colors, axis=0)


def draw_points(points, **kwargs):
    handles = []
    with pbu.LockRenderer():
        for point in points:
            handles.extend(pbu.draw_point(point, **kwargs))
    return handles


def draw_labeled_point(labeled_point, **kwargs):
    return pbu.draw_point(labeled_point.point, color=labeled_point.color, **kwargs)


def draw_labeled_points(points, **kwargs):
    handles = []
    with pbu.LockRenderer():
        for point in points:
            handles.extend(draw_labeled_point(point, **kwargs))
    return handles


def tform_labeled_points(affine, labeled_points):
    return [
        LabeledPoint(pbu.tform_point(affine, lp.point), *lp[1:])
        for lp in labeled_points
    ]


def extract_point(camera_image, pixel, world_frame=True):
    # from trimesh.scene import Camera
    rgb_image, depth_image, seg_image, camera_pose, camera_matrix = camera_image
    r, c = pixel
    height, width = depth_image.shape
    assert (0 <= r < height) and (0 <= c < width)
    # body, link = seg_image[r, c, :]
    label = seg_image if seg_image is None else seg_image[r, c]
    ray = pbu.ray_from_pixel(camera_matrix, [c, r])  # NOTE: width, height
    depth = depth_image[r, c]
    # assert not np.isnan(depth)
    point_camera = depth * ray

    point_world = pbu.tform_point(pbu.multiply(camera_pose), point_camera)
    point = (
        point_world if world_frame else point_camera
    )  # TODO: specify frame wrt the robot
    color = rgb_image[r, c, :] / MAX_PIXEL_VALUE
    return LabeledPoint(point, color, label)


def iterate_image(camera_image, step_size=3, aabb=None, **kwargs):
    if aabb is None:
        aabb = pbu.get_image_aabb(camera_image.camera_matrix)

    (height, width, _) = camera_image.rgbPixels.shape
    # TODO: clip if out of range
    (x1, y1), (x2, y2) = np.array(aabb).astype(int)
    for r in range(y1, height, step_size):
        for c in range(x1, width, step_size):
            yield pbu.Pixel(r, c)


def custom_iterate_point_cloud(
    camera_image, iterator, min_depth=0.0, max_depth=float("inf"), **kwargs
):
    rgb_image, depth_image = camera_image[:2]
    # depth_image = simulate_depth(depth_image)
    for pixel in iterator:

        depth = depth_image[pixel]
        labeled_point = extract_point(camera_image, pixel)
        if (depth <= min_depth) or (depth >= max_depth):
            continue

        yield labeled_point


def iterate_point_cloud(camera_image, **kwargs):
    return custom_iterate_point_cloud(
        camera_image, iterate_image(camera_image, **kwargs), **kwargs
    )


def extract_point_cloud(camera_image, bodies=None, **kwargs):
    if bodies is None:
        bodies = pbu.get_bodies()  # TODO avoid using pybullet objID
    labeled_points = []
    for labeled_point in iterate_point_cloud(camera_image, **kwargs):
        category, instance = labeled_point.label
        if isinstance(instance, Object) and (instance.body in bodies):
            labeled_points.append(labeled_point)
    return labeled_points


##################################################


def save_image_from_cluster(
    clusters,
    camera_image,
    directory=TEMP_DIR,
    min_points=25,
    imname="segmented_relabeledcluster.png",
):
    pbu.ensure_dir(directory)
    rgb_image, _, _, camera_pose, camera_matrix = camera_image
    relabeled_image = np.ones(rgb_image.shape[:2] + (3,))
    colors = pbu.spaced_colors(len(clusters))  # looks clearer for visualization purpose
    for i, cluster in enumerate(clusters):
        if len(cluster) < min_points:
            continue
        pixels_cluster = [
            pbu.pixel_from_ray(
                camera_matrix,
                pbu.tform_point(pbu.invert(pbu.multiply(camera_pose)), lp.point),
            ).astype(np.int32)
            for lp in cluster
        ]
        im_x, im_y = np.transpose(pixels_cluster)
        # mean_color = aggregate_color(cluster)
        relabeled_image[im_y, im_x] = colors[i]  # mean_color[:3]
    pbu.save_image(
        os.path.join(directory, imname), (relabeled_image * 255).astype(np.uint8)
    )


SPECIAL_CATEGORIES = {None: pbu.BLACK, UNKNOWN: pbu.GREY, TABLE: pbu.WHITE}


def image_from_labeled(seg_image, **kwargs):

    # TODO: order special colors
    # TODO: adjust saturation and value per category
    # labels = sorted(set(get_bodies()) | set(seg_image[..., 0].flatten()))
    labels_instance = set(seg_image[..., 1].flatten())
    detect_obj_labels = sorted(
        label
        for label in labels_instance
        if isinstance(label, str) and (label not in SPECIAL_CATEGORIES)
    )
    known_obj_labels = sorted(
        label for label in labels_instance if isinstance(label, Object)
    )  # known object in real world(table)
    labels = detect_obj_labels + known_obj_labels
    color_from_body = OrderedDict(zip(labels, pbu.spaced_colors(len(labels))))
    color_from_body.update(SPECIAL_CATEGORIES)

    image = np.zeros(seg_image.shape[:2] + (3,))
    for r in range(seg_image.shape[0]):
        for c in range(seg_image.shape[1]):
            category, instance = seg_image[r, c, :]
            if category in color_from_body:  # SPECIAL_CATEGORIES:
                color = color_from_body[category]
            else:
                color = color_from_body[instance]
            image[r, c, :] = color[:3]
    return (image * 255).astype(np.uint8)


def save_camera_images(
    camera_image, directory=utils.TEMP_DIR, prefix="", predicted=True, **kwargs
):
    # safe_remove(directory)
    pbu.ensure_dir(directory)
    rgb_image, depth_image, seg_image = camera_image[:3]
    # depth_image = simulate_depth(depth_image)
    pbu.save_image(
        os.path.join(directory, "{}rgb.png".format(prefix)), rgb_image
    )  # [0, 255]

    pbu.save_image(
        os.path.join(directory, "{}depth.png".format(prefix)), depth_image
    )  # [0, 1]
    if seg_image is None:
        return None
    if predicted:
        segmented_image = image_from_labeled(seg_image, **kwargs)
    else:
        segmented_image = pbu.image_from_segmented(seg_image)
    pbu.save_image(
        os.path.join(directory, "{}segmented.png".format(prefix)), segmented_image
    )  # [0, 255]
    return segmented_image
