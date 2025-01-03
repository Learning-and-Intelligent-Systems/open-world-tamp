import os
from collections import OrderedDict, namedtuple

import numpy as np

import owt.pb_utils as pbu
from owt.simulation.entities import TABLE, UNKNOWN, Object
from owt.utils import TEMP_DIR
from owt.voxel_utils import MAX_PIXEL_VALUE

LabeledPoint = namedtuple(
    "LabeledPoint", ["point", "color", "label"]
)  # TODO: dataclass
BACKGROUND = -1


def aggregate_color(labeled_points) -> pbu.RGBA:
    colors = [point.color for point in labeled_points]
    return pbu.RGBA(*np.median([list(c) for c in colors], axis=0))


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


def extract_point(camera_image: pbu.CameraImage, pixel: pbu.Pixel, world_frame=True):
    r, c = pixel.row, pixel.column
    height, width = camera_image.depthPixels.shape
    assert (0 <= r < height) and (0 <= c < width)
    seg_image = camera_image.segmentationMaskBuffer
    label = seg_image if seg_image is None else seg_image[r, c]
    ray = pbu.ray_from_pixel(camera_image.camera_matrix, [c, r])  # NOTE: width, height
    depth = camera_image.depthPixels[r, c]
    point_camera = depth * ray

    point_world = pbu.tform_point(pbu.multiply(camera_image.camera_pose), point_camera)
    point = point_world if world_frame else point_camera

    color = pbu.RGBA(*camera_image.rgbPixels[r, c, :] / MAX_PIXEL_VALUE)
    return LabeledPoint(point, color, label)


def iterate_image(camera_image, step_size=3, aabb=None, **kwargs):
    if aabb is None:
        aabb = pbu.get_image_aabb(camera_image.camera_matrix)

    (height, width, _) = camera_image.rgbPixels.shape

    (x1, y1), (x2, y2) = np.array([aabb.lower, aabb.upper]).astype(int)
    for r in range(y1, height, step_size):
        for c in range(x1, width, step_size):
            yield pbu.Pixel(r, c)


def custom_iterate_point_cloud(
    camera_image: pbu.CameraImage,
    iterator,
    min_depth=0.0,
    max_depth=float("inf"),
    **kwargs
):
    depth_image = camera_image.depthPixels
    for pixel in iterator:
        depth = depth_image[pixel.row, pixel.column]
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
            image[r, c, :] = [color.red, color.green, color.blue]
    return (image * 255).astype(np.uint8)


def save_camera_images(
    camera_image: pbu.CameraImage,
    directory=TEMP_DIR,
    prefix="",
    predicted=True,
    **kwargs
):
    pbu.ensure_dir(directory)
    pbu.save_image(
        os.path.join(directory, "{}rgb.png".format(prefix)), camera_image.rgbPixels
    )  # [0, 255]

    print(type(camera_image.depthPixels))
    pbu.save_image(
        os.path.join(directory, "{}depth.png".format(prefix)), camera_image.depthPixels
    )  # [0, 1]

    if camera_image.segmentationMaskBuffer is None:
        return None

    if predicted:
        segmented_image = image_from_labeled(
            camera_image.segmentationMaskBuffer, **kwargs
        )
    else:
        segmented_image = pbu.image_from_segmented(camera_image.segmentationMaskBuffer)
    pbu.save_image(
        os.path.join(directory, "{}segmented.png".format(prefix)), segmented_image
    )  # [0, 255]
    return segmented_image
