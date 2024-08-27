from __future__ import print_function

import random
import time
from collections import namedtuple
from heapq import heapify, heappop, heappush
from itertools import islice

import numpy as np

import owt.pb_utils as pbu
from owt.estimation.surfaces import get_plane_quat
from owt.simulation.control import interpolate_controller
from owt.simulation.lis import USING_ROS
from owt.simulation.utils import X_AXIS, Z_AXIS


def sample_sphere_surface(d, uniform=True):
    while True:
        v = np.random.randn(d)
        r = np.sqrt(v.dot(v))
        if not uniform or (r <= 1.0):
            return v / r


PREGRASP_DISTANCE = 0.07  # 0.05 | 0.07

# TODO: infer automatically
# FINGER_LENGTH = PR2_FINGER_DIMENSIONS[1] / 2.
FINGER_LENGTH = 0.01 if USING_ROS else 0.0  # 0. | 0.01 | 0.015 | 0.02

ScoredGrasp = namedtuple("ScoredGrasp", ["pose", "contact1", "contact2", "score"])

##################################################


def control_until_contact(
    controller, body, contact_links=[], time_after_contact=np.inf, all_contacts=True
):
    agg = all if all_contacts else any
    if (time_after_contact is np.inf) or not contact_links:
        for output in controller:
            yield output
        return
    dt = pbu.get_time_step()
    countdown = np.inf
    for output in controller:
        if countdown <= 0.0:
            break

        if (countdown == np.inf) and agg(
            pbu.single_collision(body, link=link) for link in contact_links
        ):
            countdown = time_after_contact
        yield output
        countdown -= dt


##################################################


def get_grasp(grasp_tool, gripper_from_tool=pbu.unit_pose()):
    return pbu.multiply(gripper_from_tool, grasp_tool)


def get_pregrasp(
    grasp_tool,
    gripper_from_tool=pbu.unit_pose(),
    tool_distance=PREGRASP_DISTANCE,
    object_distance=PREGRASP_DISTANCE,
):
    return pbu.multiply(
        gripper_from_tool,
        pbu.Pose(pbu.Point(x=tool_distance)),
        grasp_tool,
        pbu.Pose(pbu.Point(z=-object_distance)),
    )


##################################################


def filter_grasps(
    gripper,
    obj,
    grasp_generator,
    gripper_from_tool=pbu.unit_pose(),
    obstacles=[],
    draw=False,
    **kwargs
):
    # TODO: move to experiment.py
    obj_pose = pbu.get_pose(obj)
    for grasp_tool in grasp_generator:
        if grasp_tool is None:
            continue
        grasp_pose = pbu.multiply(
            obj_pose, pbu.invert(get_grasp(grasp_tool, gripper_from_tool))
        )
        with pbu.PoseSaver(gripper):
            pbu.set_pose(gripper, grasp_pose)  # grasp_pose | pregrasp_pose
            if any(pbu.pairwise_collision(gripper, obst) for obst in [obj] + obstacles):
                continue
            if draw:
                # print([pairwise_collision(gripper, obst) for obst in [obj] + obstacles])
                handles = pbu.draw_pose(grasp_pose)
                pbu.wait_if_gui()
                pbu.remove_handles(handles)
            yield grasp_tool


##################################################


def get_mesh_path(obj):
    info = pbu.get_model_info(obj)
    if info is None:
        [data] = pbu.get_visual_data(obj)
        path = pbu.get_data_filename(data)
    else:
        path = info.path

    return path


def mesh_from_obj(obj, use_concave=True, client=None, **kwargs):
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    [data] = pbu.get_visual_data(obj, -1, client=client)
    filename = pbu.get_data_filename(data)
    if use_concave:
        filename = filename.replace("vhacd_", "")
    scale = pbu.get_data_scale(data)
    if filename == pbu.UNKNOWN_FILE:
        raise RuntimeError(filename)
    elif filename == "":
        # Unknown mesh, approximate with bounding box
        aabb = pbu.get_aabb(obj, client=client)
        aabb_center = pbu.get_aabb_center(aabb)
        centered_aabb = pbu.AABB(
            lower=aabb.lower - aabb_center, upper=aabb.upper - aabb_center
        )
        mesh = pbu.mesh_from_points(pbu.get_aabb_vertices(centered_aabb))
    else:
        mesh = pbu.read_obj(filename, decompose=False)

    vertices = [scale * np.array(vertex) for vertex in mesh.vertices]
    vertices = pbu.tform_points(pbu.get_data_pose(data), vertices)
    return pbu.Mesh(vertices, mesh.faces)


def extract_normal(mesh, index):
    return np.array(mesh.face_normals[index, :])


def sample_grasp(
    obj,
    point1,
    point2,
    normal1,
    normal2,
    pitches=[-np.pi, np.pi],
    discrete_pitch=False,
    finger_length=FINGER_LENGTH,
    draw=False,
    **kwargs
):
    grasp_point = pbu.convex_combination(point1, point2)
    direction2 = point2 - point1
    quat = get_plane_quat(direction2)  # Biases toward the smallest rotation to align
    pitches = sorted(pitches)

    while True:
        if discrete_pitch:
            pitch = random.choice(pitches)
        else:
            pitch_range = [pitches[0], pitches[-1]]
            pitch = random.uniform(*pitch_range)
        roll = random.choice([0, np.pi])

        grasp_quat = pbu.multiply_quats(
            quat,
            pbu.quat_from_euler(pbu.Euler(roll=np.pi / 2)),
            pbu.quat_from_euler(
                pbu.Euler(pitch=pbu.pi + pitch)
            ),  # TODO: local pitch or world pitch?
            pbu.quat_from_euler(pbu.Euler(roll=roll)),  # Switches fingers
        )
        grasp_pose = pbu.Pose(grasp_point, pbu.euler_from_quat(grasp_quat))
        grasp_pose = pbu.multiply(
            grasp_pose, pbu.Pose(pbu.Point(x=finger_length))
        )  # FINGER_LENGTH

        handles = []
        if draw:
            # set_pose(gripper, multiply(grasp_pose, tool_from_root))
            draw_length = 0.05
            handles.extend(
                pbu.flatten(
                    [
                        # draw_point(grasp_point),
                        pbu.draw_point(point1, parent=obj),
                        pbu.draw_point(point2, parent=obj),
                        pbu.draw_pose(grasp_pose, length=draw_length, parent=obj),
                        [
                            pbu.add_line(point1, point2, parent=obj),
                            pbu.add_line(
                                point1,
                                point1 + draw_length * normal1,
                                parent=obj,
                                color=pbu.RED,
                            ),
                            pbu.add_line(
                                point2,
                                point2 + draw_length * normal2,
                                parent=obj,
                                color=pbu.RED,
                            ),
                        ],
                    ]
                )
            )
            # wait_if_gui() # TODO: wait_if_unlocked
            # remove_handles(handles)
        yield pbu.invert(
            grasp_pose
        ), handles  # TODO: tool_from_grasp or grasp_from_tool convention?


##################################################


def tuplify_score(s):
    if isinstance(s, tuple):
        return s
    return (s,)


def negate_score(s):
    if isinstance(s, tuple):
        return s.__class__(map(negate_score, s))
    return -s


def combine_scores(score, *scores):
    combined_score = tuplify_score(score)
    for other_score in scores:
        combined_score = combined_score + tuplify_score(other_score)
    return combined_score


def score_width(point1, point2):
    return -pbu.get_distance(point1, point2)  # Priorities small widths


def score_antipodal(error1, error2):
    return -(error1 + error2)


def score_torque(mesh, tool_from_grasp, **kwargs):
    center_mass = mesh.center_mass
    x, _, z = pbu.tform_point(tool_from_grasp, center_mass)  # Distance in xz plane
    return -pbu.get_length([x, z])


def score_overlap(
    intersector,
    point1,
    point2,
    num_samples=15,
    radius=1.5e-2,
    draw=False,
    verbose=False,
    **kwargs
):
    start_time = time.time()
    handles = []
    if draw:
        handles.append(pbu.add_line(point1, point2, color=pbu.RED))
    midpoint = np.average([point1, point2], axis=0)
    direction1 = point1 - point2
    direction2 = point2 - point1

    origins = []
    for _ in range(num_samples):
        # TODO: could return the set of surface points within a certain distance of the center
        # sample_sphere | sample_sphere_surface
        # from trimesh.sample import sample_surface_sphere
        other_direction = radius * sample_sphere_surface(
            d=3
        )  # TODO: sample rectangle for the PR2's fingers
        orthogonal_direction = np.cross(
            pbu.get_unit_vector(direction1), other_direction
        )  # TODO: deterministic
        orthogonal_direction = radius * pbu.get_unit_vector(orthogonal_direction)
        origin = midpoint + orthogonal_direction
        origins.append(origin)
        # print(get_distance(midpoint, origin))
        if draw:
            handles.append(pbu.add_line(midpoint, origin, color=pbu.RED))
    rays = list(range(len(origins)))

    direction_differences = []
    for direction in [direction1, direction2]:
        point = midpoint + direction / 2.0
        contact_distance = pbu.get_distance(midpoint, point)

        # section, slice_plane
        results = intersector.intersects_id(
            origins,
            len(origins) * [direction],  # face_indices, ray_indices, location
            return_locations=True,
            multiple_hits=True,
        )
        intersections_from_ray = {}
        for face, ray, location in zip(*results):
            intersections_from_ray.setdefault(ray, []).append((face, location))

        differences = []
        for ray in rays:
            if ray in intersections_from_ray:
                face, location = min(
                    intersections_from_ray[ray],
                    key=lambda pair: pbu.get_distance(point, pair[-1]),
                )
                distance = pbu.get_distance(origins[ray], location)
                difference = abs(contact_distance - distance)
                # normal = extract_normal(mesh, face) # TODO: use the normal for lexiographic scoring
            else:
                difference = np.nan  # INF
            differences.append(difference)
            # TODO: extract_normal(mesh, index) for the rays
        direction_differences.append(differences)

    differences1, differences2 = direction_differences
    combined = differences1 + differences2
    percent = np.count_nonzero(~np.isnan(combined)) / (len(combined))
    np.nanmean(combined)

    # return np.array([percent, -average])
    score = percent
    # score = 1e3*percent + (-average) # TODO: true lexiographic sorting
    # score = (percent, -average)

    if verbose:
        print(
            "Score: {} | Percent1: {} | Average1: {:.3f} | Percent2: {} | Average2: {:.3f} | Time: {:.3f}".format(
                score,
                np.mean(~np.isnan(differences1)),
                np.nanmean(differences1),  # nanmedian
                np.mean(~np.isnan(differences2)),
                np.nanmean(differences2),
                pbu.elapsed_time(start_time),
            )
        )  # 0.032 sec
    if draw:
        pbu.wait_if_gui()
        pbu.remove_handles(handles, **kwargs)
    return score


##################################################


def generate_mesh_grasps(
    obj,
    max_width=np.inf,
    target_tolerance=np.pi / 4,
    antipodal_tolerance=np.pi / 6,
    z_threshold=-np.inf,
    max_time=np.inf,
    max_attempts=np.inf,
    score_type="combined",
    verbose=False,
    **kwargs
):
    target_vector = pbu.get_unit_vector(Z_AXIS)

    import trimesh

    print(obj)
    vertices, faces = mesh_from_obj(obj, **kwargs)
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()

    lower, upper = pbu.AABB(*mesh.bounds)
    surface_z = lower[2]
    min_z = surface_z + z_threshold

    from trimesh.ray.ray_triangle import RayMeshIntersector

    intersector = RayMeshIntersector(mesh)

    last_time = time.time()
    attempts = last_attempts = 0
    while attempts < max_attempts:
        if (pbu.elapsed_time(last_time) >= max_time) or (last_attempts >= max_attempts):
            # break
            last_time = time.time()
            last_attempts = 0
            yield None
            continue
        attempts += 1
        last_attempts += 1

        [point1, point2], [index1, index2] = mesh.sample(2, return_index=True)
        if any(point[2] < min_z for point in [point1, point2]):
            continue
        distance = pbu.get_distance(point1, point2)
        if (distance > max_width) or (distance < 1e-3):
            continue
        direction2 = point2 - point1
        if (
            abs(pbu.angle_between(target_vector, direction2) - np.pi / 2)
            > target_tolerance
        ):
            continue

        normal1 = extract_normal(mesh, index1)
        if normal1.dot(-direction2) < 0:
            normal1 *= -1
        error1 = pbu.angle_between(normal1, -direction2)

        normal2 = extract_normal(mesh, index2)
        if normal2.dot(direction2) < 0:
            normal2 *= -1
        error2 = pbu.angle_between(normal2, direction2)

        if (error1 > antipodal_tolerance) or (error2 > antipodal_tolerance):
            continue

        # TODO: average the normals to select a new pair of contact points

        tool_from_grasp, handles = next(
            sample_grasp(obj, point1, point2, normal1, normal2, **kwargs)
        )
        tool_axis = pbu.tform_point(pbu.invert(tool_from_grasp), X_AXIS)
        if tool_axis[2] > 0:
            continue

        if score_type is None:
            score = 0.0
        elif score_type == "torque":
            score = score_torque(mesh, tool_from_grasp, **kwargs)
        elif score_type == "antipodal":
            score = score_antipodal(error1, error2, **kwargs)
        elif score_type == "width":
            score = score_width(point1, point2, **kwargs)
        elif score_type == "overlap":
            score = score_overlap(intersector, point1, point2, **kwargs)
        elif score_type == "combined":
            score = combine_scores(
                score_overlap(intersector, point1, point2, **kwargs),
                score_torque(mesh, tool_from_grasp, **kwargs),
                # score_antipodal(error1, error2),
            )
        else:
            raise NotImplementedError(score_type)

        if verbose:
            print(
                "Runtime: {:.3f} sec | Attempts: {} | Score: {}".format(
                    pbu.elapsed_time(last_time), last_attempts, score
                )
            )
        yield ScoredGrasp(tool_from_grasp, point1, point2, score)

        last_time = time.time()
        last_attempts = 0
        pbu.remove_handles(handles, **kwargs)


##################################################


def sorted_grasps(generator, max_candidates=10, p_random=0.0, **kwargs):
    candidates = []
    selected = []
    while True:
        start_time = time.time()
        for grasp in islice(generator, max_candidates - len(candidates)):
            if grasp is None:
                return
            else:
                index = len(selected) + len(candidates)
                heappush(candidates, (negate_score(grasp.score), index, grasp))
        if not candidates:
            break
        if p_random < random.random():
            score, index, grasp = candidates.pop(random.randint(0, len(candidates) - 1))
            heapify(candidates)
        else:
            score, index, grasp = heappop(candidates)
        print(
            "Grasp: {} | Index: {} | Candidates: {} | Score: {} | Time: {:.3f}".format(
                len(selected),
                index,
                len(candidates) + 1,
                negate_score(score),
                pbu.elapsed_time(start_time),
            )
        )
        yield grasp
        selected.append(grasp)


def filter_grasps(generator):
    raise NotImplementedError()
