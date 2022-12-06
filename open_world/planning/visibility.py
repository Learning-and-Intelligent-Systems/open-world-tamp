import random
import time

import numpy as np
from pybullet_tools.pr2_utils import (
    inverse_visibility,
    is_visible_point,
    set_group_conf,
    visible_base_generator,
)
from pybullet_tools.utils import (
    INF,
    PI,
    LockRenderer,
    Ray,
    batch_ray_collision,
    draw_ray,
    elapsed_time,
    get_configuration,
    get_difference,
    get_link_pose,
    get_unit_vector,
    pairwise_collision,
    point_from_pose,
    remove_handles,
    safe_zip,
    set_configuration,
    wait_if_gui,
    was_ray_hit,
)

from open_world.simulation.lis import CAMERA_OPTICAL_FRAME


def draw_rays(rays, ray_results=None, **kwargs):
    if ray_results is None:
        ray_results = [None] * len(rays)
    handles = []
    with LockRenderer():
        for ray, ray_result in safe_zip(rays, ray_results):
            if (ray_result is None) or was_ray_hit(ray_result):
                handles.extend(draw_ray(ray, ray_result, **kwargs))
    return handles


def are_points_visible(
    camera_pose, camera_matrix, points, max_depth=INF, min_depth=1e-2, draw=False
):
    start_time = time.time()
    visible = [False] * len(points)
    rays = []
    ray_indices = []
    camera_point = np.array(point_from_pose(camera_pose))
    # camera_point = tform_point(camera_pose, [0, 0, min_depth])
    for i, point in enumerate(points):
        if is_visible_point(camera_matrix, max_depth, point, camera_pose=camera_pose):
            start_point = camera_point + min_depth * get_unit_vector(
                get_difference(camera_point, point)
            )
            rays.append(Ray(start_point, point))
            ray_indices.append(i)
    ray_results = batch_ray_collision(rays)
    for i, ray_result in safe_zip(ray_indices, ray_results):
        visible[i] = not was_ray_hit(ray_result)
    print(
        "{} {} {} {:.3f}% {:.3f} sec".format(
            len(points),
            len(rays),
            sum(visible),
            sum(visible) / len(points),
            elapsed_time(start_time),
        )
    )
    if draw:
        start_time = time.time()
        handles = draw_rays(rays, ray_results)
        print("Draw time: {:.3f}".format(elapsed_time(start_time)))
        wait_if_gui()
        remove_handles(handles)
    return visible


def sample_visible_base(
    robot, target_point, obstacles=[], base_radius=1.5, theta_extent=PI / 4, **kwargs
):
    for base_conf in visible_base_generator(
        robot,
        target_point,
        base_range=(base_radius, base_radius),
        theta_range=(-theta_extent / 2, +theta_extent / 2),
        **kwargs
    ):
        # TODO: BodySaver(pr2)
        set_group_conf(robot, "base", base_conf)
        # TODO: use robot.cameras
        head_conf = inverse_visibility(
            robot, target_point, head_name=CAMERA_OPTICAL_FRAME
        )
        if head_conf is None:
            continue
        set_group_conf(robot, "head", head_conf)
        if any(pairwise_collision(robot, body) for body in obstacles):
            continue
        # yield get_group_positions(pr2)
        yield get_configuration(robot)


def optimize_visibility(
    robot,
    camera_link,
    camera_matrix,
    target_points,
    max_candidates=1,
    max_time=2,
    **kwargs
):
    start_time = time.time()
    num_candidates = 0
    best_visible, best_conf = 0, get_configuration(robot)
    while (
        (elapsed_time(start_time) < max_time)
        and (num_candidates < max_candidates)
        and (best_visible != len(target_points))
    ):
        num_candidates += 1
        sampled_point = random.choice(target_points)  # Only select on the surface
        # draw_point(sampled_point, color=GREEN)
        conf = next(sample_visible_base(robot, sampled_point, **kwargs))
        camera_pose = get_link_pose(robot, camera_link)
        visibility = are_points_visible(
            camera_pose, camera_matrix, target_points, max_depth=INF
        )
        num_visible = sum(visibility)
        if num_visible > best_visible:
            best_visible, best_conf = num_visible, conf
    print(
        "Candidates: {} | Best: {:.3f}%".format(
            num_candidates, float(best_visible) / len(target_points)
        )
    )
    set_configuration(robot, best_conf)
    return best_conf
