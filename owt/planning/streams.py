import math
import random
import time
from itertools import cycle

import numpy as np
import pybullet as p
from open_world.estimation.belief import GRASP_EXPERIMENT
from open_world.estimation.geometry import trimesh_from_body
from open_world.estimation.surfaces import z_plane
from open_world.planning.grasping import (generate_mesh_grasps, get_grasp,
                                          sorted_grasps)
from open_world.planning.primitives import (BaseSwitch, Grasp, GroupConf,
                                            GroupTrajectory, RelativePose,
                                            Sequence, Switch, Trajectory)
from open_world.planning.pushing import TOOL_POSE
from open_world.planning.samplers import (COLLISION_DISTANCE,
                                          DISABLE_ALL_COLLISIONS,
                                          MOVABLE_DISTANCE, SELF_COLLISIONS,
                                          compute_gripper_path,
                                          plan_prehensile,
                                          plan_workspace_motion,
                                          sample_attachment_base_confs,
                                          sample_prehensive_base_confs,
                                          sample_visibility_base_confs,
                                          set_open_positions,
                                          workspace_collision)
from open_world.planning.stacking import slice_mesh
from open_world.simulation.entities import WORLD_BODY, ParentBody
from pddlstream.language.constants import get_args, get_prefix
from pddlstream.utils import lowercase
from pybullet_tools.pr2_utils import (TOOL_POSE, close_until_collision,
                                      get_side_grasps, get_top_grasps,
                                      get_view_oobb)
from pybullet_tools.utils import (INF, OOBB, PI, BodySaver, Euler, Point, Pose,
                                  PoseSaver, Tuple, aabb_from_extent_center,
                                  any_link_pair_collision, buffer_aabb,
                                  compute_jacobian, convex_area,
                                  convex_combination, draw_collision_info,
                                  elapsed_time, get_aabb, get_aabb_center,
                                  get_aabb_extent, get_center_extent,
                                  get_closest_points, get_extend_fn,
                                  get_joint_names, get_joint_positions,
                                  get_length, get_link_pose,
                                  get_movable_joints, get_moving_links,
                                  get_point, get_pose, get_unit_vector,
                                  get_wrapped_pairs, inf_generator, invert,
                                  is_circular, movable_from_joints, multiply,
                                  pairwise_collision, pairwise_collisions,
                                  plan_2d_joint_motion, plan_joint_motion,
                                  point_from_pose, pose_from_tform,
                                  quat_from_pose, randomize, recenter_oobb,
                                  remove_handles, sample_placement_on_aabb,
                                  scale_aabb, set_joint_positions, set_pose,
                                  stable_z_on_aabb, tform_point, wait_if_gui,
                                  wrap_angle)

from grasp.utils import gpd_predict_grasps, graspnet_predict_grasps

SWITCH_BEFORE = "grasp"
BASE_COST = 1
PROXIMITY_COST_TERM = False
REORIENT = False
RELAX_GRASP_COLLISIONS = False

if GRASP_EXPERIMENT:
    PROXIMITY_COST_TERM = False
    REORIENT = False
    RELAX_GRASP_COLLISIONS = True

GEOMETRIC_MODES = ["top", "side", "mesh"]
LEARNED_MODES = ["gpd", "graspnet"]
MODE_ORDERS = ["", "_random", "_best"]


def close_until_collision(
    robot,
    gripper_joints,
    gripper_group,
    bodies=[],
    open_conf=None,
    closed_conf=None,
    num_steps=25,
    **kwargs
):
    if not gripper_joints:
        return None

    closed_conf, open_conf = robot.get_group_limits(gripper_group)
    resolutions = np.abs(np.array(open_conf) - np.array(closed_conf)) / num_steps
    extend_fn = get_extend_fn(robot, gripper_joints, resolutions=resolutions, **kwargs)
    close_path = [open_conf] + list(extend_fn(open_conf, closed_conf))
    collision_links = frozenset(get_moving_links(robot, gripper_joints, **kwargs))
    for i, conf in enumerate(close_path):
        set_joint_positions(robot, gripper_joints, conf, **kwargs)
        if any(
            pairwise_collision((robot, collision_links), body, **kwargs)
            for body in bodies
        ):
            if i == 0:
                return None
            return close_path[i - 1][0]
    return close_path[-1][0]


def get_grasp_candidates(robot, obj, grasp_mode="mesh", gripper_width=INF, **kwargs):
    grasp_parts = grasp_mode.split("_")
    grasp_mode = grasp_parts[0]
    if grasp_mode == "top":
        return randomize(
            get_top_grasps(
                obj,
                grasp_length=0.03,  # 0.0 | 0.03 | FINGER_LENGTH
                under=True,
                tool_pose=TOOL_POSE,
            )
        )  # get_top_cylinder_grasps
    if grasp_mode == "side":
        return randomize(
            get_side_grasps(
                obj,
                grasp_length=0.03,  # 0.0 | 0.03 | FINGER_LENGTH
                under=False,
                tool_pose=TOOL_POSE,
            )
        )  # get_side_cylinder_grasps
    if grasp_mode == "mesh":
        pitches = [-PI, PI]  # if REORIENT else [-PI / 2, 0, PI / 2]
        target_tolerance = INF if REORIENT else PI / 4  # PI / 4 | INF
        z_threshold = 1e-2  # if REORIENT else 1e-2 # -INF
        antipodal_tolerance = PI / 6  # PI / 8 | PI / 6
        generated_grasps = generate_mesh_grasps(
            obj,
            pitches=pitches,
            discrete_pitch=False,
            max_width=gripper_width,
            max_time=30,
            target_tolerance=target_tolerance,
            antipodal_tolerance=antipodal_tolerance,
            z_threshold=z_threshold,
            **kwargs
        )

        if generated_grasps is not None:
            return (
                grasp
                for grasp, contact1, contact2, score in sorted_grasps(
                    generated_grasps, max_candidates=10, p_random=0.0, **kwargs
                )
            )
        else:
            return tuple([])

    if grasp_mode in LEARNED_MODES:
        [camera] = robot.cameras
        if grasp_mode == "gpd":
            grasps_world, scores = gpd_predict_grasps(
                obj.points, camera.get_pose(), use_tool=True
            )  # world_from_tool
        elif grasp_mode == "graspnet":
            grasps_world, scores = graspnet_predict_grasps(
                obj.points, camera.get_pose()
            )  # world_from_tool
        else:
            raise NotImplementedError()
        if "random" in grasp_parts:
            grasps_world, scores = zip(*randomize(zip(grasps_world, scores)))
        if "best" in grasp_parts:
            grasps_world, scores = zip(
                *sorted(
                    zip(grasps_world, scores), key=lambda pair: pair[1], reverse=True
                )
            )
        grasps_tool = [
            multiply(invert(grasp), obj.observed_pose) for grasp in grasps_world
        ]
        return grasps_tool
        # return randomize(grasps_tool)
    raise NotImplementedError(grasp_mode)


#######################################################


def get_grasp_gen_fn(
    robot,
    other_obstacles,
    grasp_mode="mesh",
    gripper_collisions=True,
    closed_fraction=5e-2,
    max_time=60,
    max_attempts=INF,
    **kwargs
):

    grasp_mode = grasp_mode.split("_")[0]
    if grasp_mode in LEARNED_MODES:
        gripper_collisions = False

    def gen_fn(arm, obj):
        # initial_poses = {obj: get_pose(world.bodies[obj]) for obj in world.items}
        side = robot.side_from_arm(arm)
        arm_group, gripper_group, tool_name = robot.manipulators[side]
        robot.link_from_name(tool_name)
        closed_conf, open_conf = robot.get_group_limits(gripper_group)

        robot.get_group_subtree(gripper_group)
        robot.get_finger_links(robot.get_group_joints(gripper_group))
        # draw_pose(get_link_pose(robot, robot.get_group_parent(gripper_group)))
        # wait_if_gui()

        # TODO: wait until finger tip makes contact (not above links)
        # TODO: score the contact location
        set_open_positions(robot, side)
        max_width = robot.get_max_gripper_width(robot.get_group_joints(gripper_group))

        gripper = robot.get_component(gripper_group)
        parent_from_tool = robot.get_parent_from_tool(side)

        enable_collisions = gripper_collisions
        gripper_width = max_width - 1e-2  # TODO: gripper width sequence

        generator = iter(
            get_grasp_candidates(
                robot, obj, grasp_mode=grasp_mode, gripper_width=gripper_width, **kwargs
            )
        )
        last_time = time.time()
        last_attempts = 0
        while True:  # TODO: filter_grasps
            try:
                grasp_pose = next(generator)  # TODO: store past grasps
            except StopIteration:
                return
            if (
                (grasp_pose is None)
                or (elapsed_time(last_time) >= max_time)
                or (last_attempts >= max_attempts)
            ):
                if gripper_width == max_width:
                    print(
                        "Grasps for {} timed out after {} attempts and {:.3f} seconds".format(
                            obj, last_attempts, elapsed_time(last_time)
                        )
                    )
                    return
                gripper_width = max_width
                if RELAX_GRASP_COLLISIONS:
                    enable_collisions = (
                        False  # TODO: allow collisions with some parts of the gripper
                    )
                generator = iter(
                    get_grasp_candidates(
                        robot,
                        obj,
                        grasp_mode=grasp_mode,
                        gripper_width=max_width,
                        **kwargs
                    )
                )
                # yield None
                last_time = time.time()
                last_attempts = 0
                continue

            last_attempts += 1
            set_pose(obj, obj.observed_pose, **kwargs)  # TODO: hack
            set_pose(
                gripper,
                multiply(
                    get_pose(obj, **kwargs),
                    invert(get_grasp(grasp_pose, parent_from_tool)),
                ),
                **kwargs
            )
            set_joint_positions(
                gripper, robot.get_component_joints(gripper_group), open_conf, **kwargs
            )

            obstacles = []
            if enable_collisions:
                obstacles.append(obj)
            # if not REORIENT:
            obstacles.extend(other_obstacles)
            if pairwise_collisions(
                gripper, obstacles, **kwargs
            ):  # TODO: check pregrasp and/or approach path
                continue

            set_pose(
                obj, multiply(robot.get_tool_link_pose(side), grasp_pose), **kwargs
            )
            set_open_positions(robot, side)

            # wait_if_gui()
            # TODO: check collision with the resting surface (compare gripper & obj bounding boxes)
            # continue

            # if enable_collisions and any_link_pair_collision(robot, gripper_links, obj):
            if pairwise_collision(gripper, obj, **kwargs):
                continue

            set_pose(
                obj, multiply(robot.get_tool_link_pose(side), grasp_pose), **kwargs
            )
            gripper_joints = robot.get_group_joints(gripper_group)
            # TODO: individual fingers

            closed_position = closed_conf[0]
            if enable_collisions:
                closed_position = close_until_collision(
                    robot,
                    gripper_joints,
                    gripper_group,  # TODO: check only finger_links
                    bodies=[obj],
                    max_distance=0.0,
                    **kwargs
                )
                # closed_position = open_while_collision(robot, gripper_joints, bodies=[obj], max_distance=0.)
                if closed_position is None:
                    continue
                assert closed_position is not None

            if SWITCH_BEFORE in ["contact", "grasp"]:
                # closed_position = (1 - closed_fraction)*closed_position
                closed_position = (1 + closed_fraction) * closed_position
            else:
                # closed_position = closed_position
                closed_position = closed_conf[0]

            # closed_position = clip(closed_position, min_value=closed_conf[0], max_value=open_conf[0])
            # wait_if_gui()
            grasp = Grasp(obj, grasp_pose, closed_position=closed_position, **kwargs)
            print("Generated grasp after {} attempts".format(last_attempts))

            set_pose(obj, obj.observed_pose, **kwargs)
            yield Tuple(grasp)
            last_attempts = 0

    return gen_fn


#######################################################


def get_test_cfree_pose_pose(obj_obj_collisions=True, **kwargs):
    def test_cfree_pose_pose(obj1, pose1, obj2, pose2):
        if obj1 == obj2:  # or (pose2 is None): # TODO: skip if in the environment
            return True
        if obj2 in pose1.ancestors():
            return True
        pose1.assign()
        pose2.assign()
        return not pairwise_collision(obj1, obj2, max_distance=MOVABLE_DISTANCE)

    return test_cfree_pose_pose


def get_cfree_pregrasp_pose_test(robot, **kwargs):

    def test(arm, obj1, pose1, grasp1, obj2, pose2):
        side = robot.side_from_arm(arm)
        if obj1 == obj2:  # or (pose2 is None):
            return True
        if obj2 in pose1.ancestors():
            return True
        if (pose1.important and not obj1.is_fragile) and (
            pose2.important and not obj2.is_fragile
        ):
            return True
        pose2.assign()
        gripper_path = compute_gripper_path(pose1, grasp1)
        grasp = None if (pose1.important and pose2.important) else grasp1
        return not workspace_collision(
            robot,
            side,
            gripper_path,
            grasp,
            obstacles=[obj2],
            max_distance=MOVABLE_DISTANCE,
        )

    return test


def get_cfree_traj_pose_test(robot, **kwargs):
    def test(arm, sequence, obj2, pose2):
        if pose2 is None:  # (obj1 == obj2) or
            return True
        # if obj2 in pose1.ancestors():
        #     return True
        if sequence.name.startswith("pick") and (
            pose2.important and not obj2.is_fragile
        ):
            return True
        if obj2 in sequence.context_bodies:
            return True
        pose2.assign()
        set_open_positions(robot, arm)
        # state = State() # TODO: apply to the context

        for traj in sequence.commands:
            if not isinstance(traj, GroupTrajectory):
                continue
            if obj2 in traj.context_bodies:  # TODO: check the grasp
                continue
            moving_links = get_moving_links(traj.robot, traj.joints)
            # for _ in command.iterate(state=None):
            for _ in traj.traverse():
                # wait_if_gui()
                if any_link_pair_collision(
                    traj.robot, moving_links, obj2, max_distance=MOVABLE_DISTANCE
                ):  # \
                    # or any_link_pair_collision(traj.robot, moving_links, other_target_link, max_distance=MOVABLE_DISTANCE):
                    return False
        return True

    return test


#######################################################

# def get_default_poses(obj, weight=1.):
#     default_pose = Pose()
#     yield default_pose, weight


def compute_stable_poses(obj, weight=0.5, min_prob=0.0, min_area=None):
    # TODO: filter similar orientations (if only a change in yaw)
    # from trimesh.path.packing import paths, polygons, rectangles
    default_pose = Pose()
    yield default_pose, weight
    if weight >= 1:
        return
    history = [default_pose]
    obj_trimesh = trimesh_from_body(obj)
    set_pose(obj, default_pose)

    pose_mats, poses_prob = obj_trimesh.compute_stable_poses(
        center_mass=None, sigma=0.0, n_samples=1, threshold=min_prob
    )
    # pose_mats = pose_mats[poses_prob>.1]
    for pose_mat, pose_score in list(zip(pose_mats, poses_prob)):  # reversed
        area = np.nan
        if min_area is not None:
            new_trimesh = obj_trimesh.copy().apply_transform(
                pose_mat
            )  # apply_transform modifies the input mesh
            # print(new_trimesh.bounds)
            surfaces = slice_mesh(new_trimesh, plane=z_plane(z=1e-2))
            if not surfaces:
                continue
            surface = surfaces[0]
            area = 0.0 if surface is None else convex_area(surface.vertices)
            if area <= min_area:
                continue

        print(
            "Num: {} | Prob: {:.3f} | Area: {:.3f}".format(
                len(history), pose_score, area
            )
        )
        top_pose = pose_from_tform(pose_mat)
        set_pose(obj, top_pose)
        offset_center1, offset_extent1 = get_center_extent(obj)
        offset_center2 = get_point(obj)
        dz = (offset_center1[2] - offset_extent1[2] / 2) - offset_center2[2]
        top_pose = (top_pose[0] + (0, 0, dz), top_pose[1])
        set_pose(obj, top_pose)

        # print(len(history), euler_from_quat(top_pose[1]), pose_score, scores)
        yield top_pose, (1 - weight) * pose_score
        history.append(top_pose)
        # TODO: compare orientation similarity in order to prune similar


def generate_stable_poses(obj, deterministic=False, **kwargs):
    # TODO: place on the most stable face
    # TODO: placement that maximizes height
    # TODO: place cost dependent on the quality of the placement
    start_time = time.time()
    weight = 0.0 if REORIENT else 1.0
    # weight = 0.5 if REORIENT else 1.
    poses, scores = zip(*compute_stable_poses(obj, weight=weight, **kwargs))
    print(
        "Poses: {} | Scores: {} | Time: {:.3f}".format(
            len(poses), np.round(scores, 3).tolist(), elapsed_time(start_time)
        )
    )
    if deterministic:
        generator = cycle(iter(poses))
    else:
        # TODO: unweighted version of this if above a threshold
        generator = (
            random.choices(poses, weights=scores, k=1)[0] for _ in inf_generator()
        )  # TODO: python2
    return generator


#######################################################


def get_placement_gen_fn(
    robot,
    other_obstacles,
    environment=[],
    buffer=2e-2,
    max_distance=INF,
    max_attempts=100,
    **kwargs
):  # max_distance=PR2_WINGSPAN
    base_pose = get_link_pose(robot, robot.base_link, **kwargs)

    def gen_fn(obj, surface, surface_pose):
        surface_pose.assign()
        surface_oobb = (
            surface.get_shape_oobb()
        )  # TODO: change to as long as the COM is on
        # draw_oobb(surface_oobb)
        obstacles = set(environment) - {obj, surface}  # TODO: surface might have walls

        aabb = surface_oobb.aabb
        # aabb = buffer_aabb(aabb, buffer)
        aabb = aabb_from_extent_center(
            2 * buffer * np.array([1, 1, 0]) + get_aabb_extent(aabb),
            get_aabb_center(aabb),
        )
        for top_pose in generate_stable_poses(obj):  # cycle
            pose = sample_placement_on_aabb(
                obj,
                aabb,
                max_attempts=max_attempts,
                top_pose=top_pose,  # TODO: reference pose instead?
                percent=1.0,
                epsilon=1e-3,
                **kwargs
            )  # TODO: Z_EPSILON
            if pose is None:
                # yield None
                continue
            pose = multiply(surface_oobb.pose, pose)
            set_pose(obj, pose, **kwargs)
            rel_pose = RelativePose(
                obj,
                parent=ParentBody(surface, **kwargs),
                parent_state=surface_pose,
                **kwargs
            )  # , relative_pose=pose)
            base_distance = get_length(
                point_from_pose(multiply(invert(base_pose), rel_pose.get_pose()))[:2]
            )
            if (surface in other_obstacles) and (base_distance > max_distance):
                continue
            if pairwise_collisions(
                obj, obstacles - set(rel_pose.ancestors()), max_distance=0.0, **kwargs
            ):
                # TODO: max_attempts here as well
                continue
            yield Tuple(rel_pose)
        # yield None

    return gen_fn


def get_mobile_placement_gen_fn(
    robot,
    other_obstacles,
    environment=[],
    floor=None,
    buffer=2e-2,
    max_distance=INF,
    max_attempts=100,
    **kwargs
):  # max_distance=PR2_WINGSPAN
    get_link_pose(robot, robot.base_link, **kwargs)

    def gen_fn(obj):
        floor_pose = RelativePose(floor, important=True, **kwargs)
        surface_oobb = (
            floor.get_shape_oobb()
        )  # TODO: change to as long as the COM is on
        obstacles = set(environment) - {obj, floor}  # TODO: surface might have walls
        aabb = surface_oobb.aabb
        aabb = aabb_from_extent_center(
            2 * buffer * np.array([1, 1, 0]) + get_aabb_extent(aabb),
            get_aabb_center(aabb),
        )
        for top_pose in generate_stable_poses(obj):  # cycle
            pose = sample_placement_on_aabb(
                obj,
                aabb,
                max_attempts=max_attempts,
                top_pose=top_pose,  # TODO: reference pose instead?
                percent=1.0,
                epsilon=1e-3,
                **kwargs
            )  # TODO: Z_EPSILON
            if pose is None:
                # yield None
                continue
            pose = multiply(surface_oobb.pose, pose)
            set_pose(obj, pose, **kwargs)
            rel_pose = RelativePose(
                obj,
                parent=ParentBody(floor, **kwargs),
                parent_state=floor_pose,
                **kwargs
            )  # , relative_pose=pose)
            yield Tuple(rel_pose)
        # yield None

    return gen_fn


def get_pose_cost_fn(robot, cost_per_m=1.0, **kwargs):
    # TODO(caelan): refactor
    base_pose = get_link_pose(robot, robot.base_link, **kwargs)

    def cost_fn(obj, pose):
        cost = BASE_COST
        if PROXIMITY_COST_TERM:  # Closest is least costly
            point_base = tform_point(
                invert(base_pose), point_from_pose(pose.get_pose())
            )
            distance = get_length(point_base[:2])
            cost += cost_per_m * distance
        if GRASP_EXPERIMENT:
            cost += random.random()  # Break ties for holding any object
        return cost

    return cost_fn


def get_cardinal_sample(robot, direction, **kwargs):

    def fn(obj1, pose, obj2):

        pose1 = pose.get_pose()

        if direction == "leftof":
            set_pose(obj2, [(pose1[0][0] - 0.2, pose1[0][1], pose1[0][2]), pose1[1]])
        if direction == "rightof":
            set_pose(obj2, [(pose1[0][0] + 0.2, pose1[0][1], pose1[0][2]), pose1[1]])
        if direction == "aheadof":
            set_pose(obj2, [(pose1[0][0], pose1[0][1] + 0.2, pose1[0][2]), pose1[1]])
        if direction == "behind":
            set_pose(obj2, [(pose1[0][0], pose1[0][1] - 0.2, pose1[0][2]), pose1[1]])

        rel_pose = RelativePose(
            obj2, parent=ParentBody(obj1, **kwargs), parent_state=pose1, **kwargs
        )  # , relative_pose=pose)

        return Tuple(rel_pose)

    return fn


def get_reachability_test(
    robot, step_size=1e-1, max_iterations=200, draw=False, **kwargs
):
    robot_saver = BodySaver(robot, client=robot.client)

    def test(arm, obj, pose, base_conf):
        side = robot.side_from_arm(arm)
        arm_group, gripper_group, tool_name = robot.manipulators[side]
        target_link = robot.get_group_parent(gripper_group)
        arm_joints = robot.get_group_joints(arm_group)

        robot_saver.restore()
        pose.assign()
        base_conf.assign()

        # TODO: average over multiple links (e.g. finger tips)
        # TODO: adjust the orientation to point
        success = False
        start_time = time.time()
        for iteration in range(max_iterations):
            current_conf = get_joint_positions(robot, arm_joints)
            collision_infos = get_closest_points(
                robot, obj, link1=target_link, max_distance=INF
            )  # tool_link
            handles = []
            if draw:
                for collision_info in collision_infos:
                    handles.extend(draw_collision_info(collision_info))
            [collision_info] = collision_infos
            distance = collision_info.contactDistance
            # TODO: convergence or decay in step size
            direction = -step_size * get_unit_vector(
                collision_info.contactNormalOnB
            )  # Already normalized
            # print('Iteration: {} | Collisions: {} | Distance: {:.3f} | Time: {:.3f}'.format(
            #    iteration, len(collision_infos), distance, elapsed_time(start_time)))
            if distance <= 0.0:
                success = True
                break
            translate, _ = compute_jacobian(robot, target_link)
            correction_conf = np.array(
                [
                    np.dot(translate[mj], direction)
                    for mj in movable_from_joints(robot, arm_joints)
                ]
            )
            set_joint_positions(robot, arm_joints, current_conf + correction_conf)
            if draw:
                wait_if_gui()
            remove_handles(handles)
        print(
            "Reachability: {} | Iteration: {} | Time: {:.3f}".format(
                success, iteration, elapsed_time(start_time)
            )
        )
        return success

    return test


#######################################################


def get_plan_mobile_attach_fn(robot, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)

    def fn(obj, pose):
        robot_saver.restore()
        base_conf = next(sample_attachment_base_confs(robot, obj, pose, **kwargs))
        base_pose = Pose(
            point=Point(x=base_conf.positions[0], y=base_conf.positions[1]),
            euler=Euler(yaw=base_conf.positions[2]),
        )
        base_grasp = multiply(invert(base_pose), pose.get_pose())

        switch = BaseSwitch(
            obj,
            parent=ParentBody(
                body=robot, link=robot.link_from_name("base_chassis_link")
            ),
        )
        commands = [switch]
        sequence = Sequence(commands=commands, name="attach-{}".format(obj))
        return Tuple(base_conf, base_grasp, sequence)

    return fn


def get_plan_mobile_detach_fn(robot, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)

    def fn(obj, pose, base_grasp):
        # TODO: generator instead of a function
        robot_saver.restore()
        base_pose = multiply(pose.get_pose(), invert(base_grasp))
        base_point, base_quat = base_pose
        base_euler = p.getEulerFromQuaternion(base_quat)
        base_positions = robot.get_default_conf()[robot.base_group]
        base_positions[:3] = [base_point[0], base_point[1], base_euler[2]]
        base_conf = GroupConf(
            robot, robot.base_group, positions=base_positions, **kwargs
        )
        switch = BaseSwitch(obj, parent=WORLD_BODY)

        # TODO: wait for a bit and remove colliding objects
        commands = [switch]
        sequence = Sequence(commands=commands, name="place-{}".format(obj))
        return Tuple(base_conf, sequence)

    return fn


def get_plan_pick_fn(robot, environment=[], **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)
    environment = environment

    def fn(arm, obj, pose, grasp, base_conf):
        # TODO: generator instead of a function
        # TODO: add the ancestors as collision obstacles
        robot_saver.restore()
        base_conf.assign()
        arm_path = plan_prehensile(robot, arm, obj, pose, grasp, **kwargs)

        if arm_path is None:
            return None

        arm_group, gripper_group, tool_name = robot.manipulators[
            robot.side_from_arm(arm)
        ]
        arm_traj = GroupTrajectory(
            robot,
            arm_group,
            arm_path[::-1],
            context=[pose],
            velocity_scale=0.25,
            client=robot.client,
        )
        arm_conf = arm_traj.first()

        closed_conf = grasp.closed_position * np.ones(
            len(robot.get_group_joints(gripper_group))
        )
        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[closed_conf],
            contexts=[pose],
            contact_links=robot.get_finger_links(robot.get_group_joints(gripper_group)),
            time_after_contact=1e-1,
            client=robot.client,
        )
        switch = Switch(
            obj,
            parent=ParentBody(
                body=robot, link=robot.link_from_name(tool_name), client=robot.client
            ),
        )

        # TODO: close the gripper a little bit before pregrasp
        if SWITCH_BEFORE == "contact":
            commands = [arm_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "grasp":
            commands = [arm_traj, switch, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "pregrasp":
            commands = [arm_traj, gripper_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "arm":
            commands = [arm_traj, gripper_traj, arm_traj.reverse(), switch]
        elif SWITCH_BEFORE == "none":
            commands = [arm_traj, gripper_traj, arm_traj.reverse()]
        else:
            raise NotImplementedError(SWITCH_BEFORE)

        sequence = Sequence(
            commands=commands, name="pick-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn


def get_plan_mobile_pick_fn(robot, max_attempts=5, **kwargs):
    from open_world.planning.drake.drake_controller import drake_ik

    robot_saver = BodySaver(robot, client=robot.client)
    pick_fn = get_plan_pick_fn(robot, **kwargs)

    def fn(arm, obj, pose, grasp):

        while True:
            robot_saver.restore()

            movable_joints = get_movable_joints(robot, client=robot.client)
            q0 = get_joint_positions(robot, movable_joints, client=robot.client)
            movable_joint_names = get_joint_names(
                robot, movable_joints, client=robot.client
            )
            pose.assign()
            gripper_path = compute_gripper_path(pose, grasp)
            gripper_pose = gripper_path[0]
            positions = None

            num_attempts = 0
            while positions is None:
                positions = drake_ik(
                    gripper_pose[0], gripper_pose[1], movable_joint_names, q0
                )
                num_attempts += 1
                if num_attempts > max_attempts:
                    return None
            position_dict = {
                joint_name: joint_value
                for joint_name, joint_value in zip(movable_joint_names, positions)
            }
            arm_group_joints = robot.get_group_joints(arm)
            arm_joint_names = get_joint_names(
                robot, arm_group_joints, client=robot.client
            )

            base_group_joints = robot.get_group_joints(robot.base_group)
            base_joint_names = get_joint_names(
                robot, base_group_joints, client=robot.client
            )

            arm_positions = [
                position_dict[joint_name] for joint_name in arm_joint_names
            ]
            base_positions = [
                position_dict[joint_name] for joint_name in base_joint_names
            ]

            arm_conf = GroupConf(
                robot, arm, positions=arm_positions, client=robot.client
            )
            base_conf = GroupConf(
                robot, robot.base_group, positions=base_positions, client=robot.client
            )

            yield Tuple(base_conf, arm_conf)

    return fn


#######################################################


def get_plan_place_fn(robot, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)

    def fn(arm, obj, pose, grasp, base_conf):
        # TODO: generator instead of a function
        robot_saver.restore()
        base_conf.assign()
        arm_path = plan_prehensile(robot, arm, obj, pose, grasp, **kwargs)
        if arm_path is None:
            return None

        arm_group, gripper_group, tool_name = robot.manipulators[
            robot.side_from_arm(arm)
        ]
        arm_traj = GroupTrajectory(
            robot,
            arm_group,
            arm_path[::-1],
            context=[grasp],
            velocity_scale=0.25,
            client=robot.client,
        )
        arm_conf = arm_traj.first()

        closed_conf, open_conf = robot.get_group_limits(gripper_group)
        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[open_conf],
            contexts=[grasp],
            client=robot.client,
        )
        switch = Switch(obj, parent=WORLD_BODY)

        # TODO: wait for a bit and remove colliding objects
        if SWITCH_BEFORE == "contact":
            commands = [arm_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "grasp":
            commands = [arm_traj, gripper_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "pregrasp":
            commands = [arm_traj, switch, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "arm":
            commands = [switch, arm_traj, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "none":
            commands = [arm_traj, gripper_traj, arm_traj.reverse()]
        else:
            raise NotImplementedError(SWITCH_BEFORE)
        sequence = Sequence(
            commands=commands, name="place-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn


def get_plan_mobile_place_fn(robot, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)
    place_fn = get_plan_place_fn(robot, **kwargs)

    def fn(arm, obj, pose, grasp):
        robot_saver.restore()
        for base_conf in sample_prehensive_base_confs(
            robot, arm, obj, pose, grasp, **kwargs
        ):
            outputs = place_fn(arm, obj, pose, grasp, base_conf)
            if outputs is None:
                continue
            yield Tuple(base_conf) + outputs

    return fn


def get_plan_mobile_look_fn(
    robot, environment=[], max_head_attempts=10, max_base_attempts=100, **kwargs
):
    robot_saver = BodySaver(robot, client=robot.client)

    def fn(obj, pose):
        while True:
            robot_saver.restore()
            # TODO: Return a head conf that will lead to visibility of obj at pose
            if robot.head_group is None:
                return None
            else:
                pose.assign()
                limits = list(robot.get_group_limits(robot.head_group))
                num_base_attempts = 0
                for base_conf in sample_visibility_base_confs(
                    robot, obj, pose, client=robot.client
                ):
                    visible = False
                    base_conf.assign()
                    num_head_attempts = 0
                    while not visible:
                        random_head_pos = [
                            random.uniform(*limit) for limit in zip(*limits)
                        ]
                        robot.set_group_positions(robot.head_group, random_head_pos)
                        visible = robot.cameras[0].object_visible(obj)
                        num_head_attempts += 1
                        if num_head_attempts >= max_head_attempts:
                            break
                    if num_head_attempts >= max_head_attempts:
                        continue
                    gp = random_head_pos
                    current_hq = GroupConf(
                        robot, robot.head_group, gp, client=robot.client
                    )
                    num_base_attempts += 1

                    yield (base_conf, current_hq)
                    if num_base_attempts > max_base_attempts:
                        return None
            num_attempts += 1

    return fn


def get_plan_look_fn(robot, environment=[], max_attempts=1000, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)

    def fn(obj, pose, base_conf):
        while True:
            robot_saver.restore()
            base_conf.assign()
            # TODO: Return a head conf that will lead to visibility of obj at pose
            if robot.head_group is None:
                return None
            else:
                pose.assign()
                visible = False
                limits = list(robot.get_group_limits(robot.head_group))
                num_attempts = 0
                while not visible:
                    random_head_pos = [random.uniform(*limit) for limit in zip(*limits)]
                    robot.set_group_positions(robot.head_group, random_head_pos)
                    visible = robot.cameras[0].object_visible(obj)
                    num_attempts += 1
                    if num_attempts >= max_attempts:
                        num_attempts = 0
                        yield None

                gp = random_head_pos
                current_hq = GroupConf(robot, robot.head_group, gp, client=robot.client)
                yield (current_hq,)
            num_attempts += 1

    return fn


def get_plan_drop_fn(robot, environment=[], z_offset=2e-2, shrink=0.25, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)

    def fn(arm, obj, grasp, bin, bin_pose, base_conf):
        # TODO: don't necessarily need the grasp
        robot_saver.restore()
        base_conf.assign()
        bin_pose.assign()
        obstacles = list(environment)

        side = robot.side_from_arm(arm)
        _, gripper_group, _ = robot.manipulators[side]
        gripper = robot.get_component(gripper_group)
        parent_from_tool = robot.get_parent_from_tool(side)

        bin_aabb = get_aabb(bin)
        # _, (_, _, z) = bin_aabb
        # x, y, _ = get_aabb_center(bin_aabb)
        # gripper_pose = Pose(point=Point(x, y, z + 0.1), euler=DOWNWARD_EULER)

        # reference_pose = unit_pose()
        reference_pose = multiply(
            Pose(euler=Euler(pitch=PI / 2, yaw=random.uniform(0, 2 * PI))), grasp.value
        )
        # obj_pose = sample_placement_on_aabb(obj, bin_aabb, top_pose=reference_pose, percent=shrink, epsilon=1e-2)
        # _, extent = approximate_as_prism(obj, reference_pose=reference_pose)
        with PoseSaver(obj):
            set_pose(obj, reference_pose)
            obj_pose = (
                np.append(
                    get_aabb_center(bin_aabb)[:2],
                    [stable_z_on_aabb(obj, bin_aabb) + z_offset],
                ),
                quat_from_pose(reference_pose),
            )  # TODO: get_aabb_top, get_aabb_bottom

        if obj_pose is None:
            return None
        gripper_pose = multiply(obj_pose, invert(grasp.value))
        set_pose(gripper, multiply(gripper_pose, invert(parent_from_tool)))
        set_pose(obj, multiply(gripper_pose, grasp.value))
        if any(
            pairwise_collisions(body, environment, max_distance=0.0)
            for body in [obj, gripper]
        ):
            return None

        _, _, tool_name = robot.manipulators[robot.side_from_arm(arm)]
        attachment = grasp.create_attachment(
            robot, link=robot.link_from_name(tool_name)
        )

        arm_path = plan_workspace_motion(
            robot, side, [gripper_pose], attachment=attachment, obstacles=obstacles
        )
        if arm_path is None:
            return None
        arm_conf = GroupConf(robot, arm, positions=arm_path[0], **kwargs)
        switch = Switch(obj, parent=WORLD_BODY)

        closed_conf, open_conf = robot.get_group_limits(gripper_group)
        # gripper_joints = robot.get_group_joints(gripper_group)
        # closed_conf = grasp.closed_position * np.ones(len(gripper_joints))
        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[closed_conf, open_conf],
            contexts=[],
            client=robot.client,
        )

        commands = [switch, gripper_traj]
        sequence = Sequence(
            commands=commands, name="drop-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn


#######################################################


def get_plan_motion_fn(
    robot, environment=[], **kwargs
):  # , collisions=True): #, teleport=False):
    robot_saver = BodySaver(robot, client=robot.client)
    robot_aabb = scale_aabb(recenter_oobb(robot.get_shape_oobb()).aabb, 0.5)

    def fn(group, q1, q2, fluents=[]):
        robot_saver.restore()
        print("Plan motion fn {}->{}".format(q1, q2))
        # if q1 == q2:
        #     return None
        obstacles = list(environment)
        attachments = []
        base_attachments = []
        for fluent in fluents:
            predicate = get_prefix(fluent)
            args = get_args(fluent)
            if predicate in lowercase("AtConf"):
                args[-1].assign()
            elif predicate in lowercase("AtPose"):
                body, pose = args
                if pose is None:
                    continue

                if body.get_shape_oobb().aabb.upper[2] < 0.01:
                    # Filter out the floor
                    continue
                obstacles.append(body)
                pose.assign()
            elif predicate in lowercase("AtGrasp"):
                arm, body, grasp = args
                # if not body.is_fragile:
                #     # TODO: remove from the scene as a collision object
                #     continue
                side = robot.get_arbitrary_side()
                _, _, tool_name = robot.manipulators[side]
                tool_link = robot.link_from_name(tool_name)
                attachment = grasp.create_attachment(robot, link=tool_link)
                attachment.assign()
                attachments.append(attachment)
            elif predicate in lowercase("AtAttachmentGrasp"):
                body, grasp = args
                # Get object pose in base frame
                base_attachments.append((get_aabb(body, client=robot.client), grasp))
            else:
                raise NotImplementedError(predicate)
        attached = {attachment.child for attachment in attachments}
        obstacles = set(obstacles) - attached
        q1.assign()

        # TODO: separate collision resolution for movable
        if group == robot.base_group:
            resolutions = 0.1 * np.ones(len(q2.joints))
            min_vals, max_vals = robot.get_group_limits(robot.base_group)
            path = plan_2d_joint_motion(
                robot,
                robot_aabb,
                q2.joints,
                min_vals,
                max_vals,
                q1.positions,
                q2.positions,
                resolutions=resolutions,
                obstacle_oobbs=[obstacle.get_shape_oobb() for obstacle in obstacles],
                restarts=0,
                max_iterations=100,
                smooth=100,
                attachments=base_attachments,
                disable_collisions=DISABLE_ALL_COLLISIONS,
                **kwargs
            )
            print("Output path: " + str(path))
        else:
            resolutions = math.radians(10) * np.ones(len(q2.joints))

            path = plan_joint_motion(
                robot,
                q2.joints,
                q2.positions,
                resolutions=resolutions,
                # weights=weights, # TODO: joint weights
                obstacles=obstacles,
                attachments=attachments,
                self_collisions=SELF_COLLISIONS,
                disabled_collisions=robot.disabled_collisions,
                max_distance=COLLISION_DISTANCE,
                custom_limits=robot.custom_limits,
                restarts=1,
                iterations=5,
                smooth=100,
                disable_collisions=DISABLE_ALL_COLLISIONS,
                **kwargs
            )

        if path is None:
            for conf in [q1, q2]:
                conf.assign()
                for attachment in attachments:
                    attachment.assign()
            return None

        sequence = Sequence(
            commands=[
                GroupTrajectory(robot, group, path, client=robot.client),
            ],
            name="move-{}".format(group),
        )
        return Tuple(sequence)

    return fn


#######################################################


def get_nominal_test(robot, side="left", axis=1, **kwargs):
    def gen_fn(obj, pose, region, region_pose):
        value = point_from_pose(pose.relative_pose)[axis]
        success = (value > 0) if side == "left" else (value < 0)
        return success

    return gen_fn
