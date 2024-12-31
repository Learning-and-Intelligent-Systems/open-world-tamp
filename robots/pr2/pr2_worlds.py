import itertools
import math
import random

import numpy as np
import pybullet as p

import owt.pb_utils as pbu
from owt.simulation.entities import RealWorld
from owt.simulation.environment import (Pose2D, create_bin, create_cubbies,
                                        create_cubby, create_floor_object,
                                        create_object, create_pillar,
                                        create_table_object, create_tray,
                                        create_ycb, get_grid_cells,
                                        place_object, place_surface)


def create_default_env(robot, **kwargs):
    # pbu.set_pose(robot, pbu.Pose(pbu.Point(x=-0.5)), **kwargs)

    pbu.set_camera_pose(
        camera_point=[0.75, -0.75, 1.25], target_point=[-0.75, 0.75, 0.0], **kwargs
    )
    pbu.draw_pose(pbu.Pose(), length=1, **kwargs)

    pbu.add_data_path()
    with pbu.HideOutput(enable=True):
        create_floor_object(**kwargs)
        table = create_table_object(**kwargs)

        pbu.set_pose(table, pbu.Pose(pbu.Point(x=0.5)), **kwargs)
        obstacles = [table]

        for obst in obstacles:
            pbu.set_dynamics(
                obst,
                lateralFriction=1.0,  # linear (lateral) friction
                spinningFriction=1.0,  # torsional friction around the contact normal
                rollingFriction=0.01,  # torsional friction orthogonal to contact normal
                restitution=0.0,  # restitution: 0 => inelastic collision, 1 => elastic collision
                **kwargs
            )

    return table, obstacles


def create_world(robot, movable=[], fixed=[], surfaces=[], **kwargs):
    obstacles = sorted(set(fixed) | set(surfaces))
    return RealWorld(
        robot,
        movable=movable,
        fixed=obstacles,
        detectable=movable,
        known=obstacles,
        surfaces=surfaces,
        **kwargs
    )


#######################################################


def problem0(args, robot, **kwargs):
    table, obstacles = create_default_env(robot, **kwargs)
    region = place_surface(
        create_pillar(width=0.3, length=0.3, color=pbu.GREEN, **kwargs),
        table,
        yaw=np.pi / 4,
        **kwargs
    )

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=np.pi / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region], **kwargs
    )

    return real_world


def create_pybullet_block(
    color, half_extents, mass, friction, orientation, client=None
):
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    pose = (0, 0, 0)

    # Create the collision shape.
    collision_id = client.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)

    # Create the visual_shape.
    visual_id = client.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )

    # Create the body.
    block_id = client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=pose,
        baseOrientation=orientation,
    )
    client.changeDynamics(
        block_id, linkIndex=-1, lateralFriction=friction  # -1 for the base
    )

    return block_id


def problem_five_blocks(args, robot, **kwargs):
    arms = args.arms  # ARM_NAMES
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)

    table_width = 0.6
    table_length = 1.2
    table_height = 0.73
    table_pose = pbu.get_pose(table)[0]

    block_size = 0.045
    _obj_colors = [
        (0.95, 0.05, 0.1, 1.0),
        (0.05, 0.95, 0.1, 1.0),
        (0.1, 0.05, 0.95, 1.0),
        (0.95, 0.95, 0.1, 1.0),
        (0.1, 0.1, 0.1, 1.0),
    ]

    # Object parameters.
    _obj_mass = 0.5
    _obj_friction = 1.2
    _default_orn = [0.0, 0.0, 0.0, 1.0]

    objs = []
    for i in range(5):
        color = _obj_colors[i % len(_obj_colors)]
        half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)
        objs.append(
            create_pybullet_block(
                color, half_extents, _obj_mass, _obj_friction, _default_orn, **kwargs
            )
        )

    for block_index, block_id in enumerate(objs):
        found_collision_free = False
        timeout = 100
        padding = 0.1
        while not found_collision_free or timeout > 0:
            timeout -= 1
            rx = random.uniform(
                table_pose[0] - table_width / 2.0 + padding,
                table_pose[0] + table_width / 2.0 - padding,
            )
            ry = random.uniform(
                table_pose[1] - table_length / 2.0 + padding,
                table_pose[1] + table_length / 2.0 - padding,
            )
            pbu.set_pose(
                block_id,
                ([rx, ry, table_height + block_size / 2.0], _default_orn),
                **kwargs
            )
            collision = False
            for placed_block in objs[:block_index]:
                if pbu.pairwise_collision(block_id, placed_block, **kwargs):
                    collision = True
                    break
            if not collision:
                break

    real_world = create_world(
        robot, movable=[objs], fixed=obstacles, surfaces=[table], **kwargs
    )

    return real_world


def problem1(args, robot, **kwargs):
    arms = args.arms  # ARM_NAMES
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(
        create_pillar(width=0.3, length=0.3, color=pbu.GREEN, **kwargs),
        table,
        yaw=np.pi / 4,
        **kwargs
    )

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=np.pi / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region], **kwargs
    )

    return real_world


def full_occlusion_mem(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(
        create_pillar(width=0.18, color=pbu.GREEN, **kwargs), table, **kwargs
    )

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can"), table, Pose2D(yaw=np.pi / 2, x=0.07), **kwargs
    )
    obj2 = place_object(
        create_ycb("cracker_box", **kwargs), table, Pose2D(x=0), **kwargs
    )
    obj3 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=np.pi / 4, y=0.4), **kwargs
    )
    place_object_rotate(
        obj3, table, x=0.08, y=0.4, yaw=np.pi / 2, roll=np.pi / 2, **kwargs
    )
    real_world = create_world(
        robot,
        movable=[obj1, obj2, obj3],
        fixed=obstacles,
        surfaces=[table, region],
        **kwargs
    )
    return real_world


def full_occlusion(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(create_pillar(color=pbu.GREEN, **kwargs), table, **kwargs)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can"), table, Pose2D(yaw=np.pi / 2, x=0.07)
    )
    obj2 = place_object(create_ycb("cracker_box"), table, Pose2D(x=0))
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def problem1_non_convex(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(create_pillar(color=pbu.GREEN, **kwargs), table, **kwargs)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can", use_concave=True, **kwargs),
        table,
        Pose2D(yaw=np.pi / 4),
        **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region], **kwargs
    )
    assert not args.convex

    return real_world


def problem1_color(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(create_pillar(color=pbu.GREEN, **kwargs), table, **kwargs)

    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=np.pi / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def problem2(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(
        create_pillar(width=0.15, length=0.15, color=pbu.GREEN, **kwargs),
        table,
        **kwargs
    )

    obj1 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=np.pi / 4), **kwargs
    )
    obj2 = place_object(create_ycb("tomato_soup_can"), region, Pose2D(), **kwargs)
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table, region]
    )
    return real_world


def problem3(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    red_region = place_surface(
        create_pillar(width=0.15, length=0.15, height=0.05, color=pbu.RED), table, y=0.2
    )
    blue_region = place_surface(
        create_pillar(width=0.15, length=0.15, height=0.05, color=pbu.BLUE),
        table,
        y=0.4,
    )

    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=np.pi / 4)
    )
    obj2 = place_object(
        create_ycb("tomato_soup_can"), table, Pose2D(y=-0.2, yaw=np.pi / 8)
    )
    real_world = create_world(
        robot,
        movable=[obj1, obj2],
        fixed=obstacles,
        surfaces=[table, red_region, blue_region],
    )

    return real_world


def place_tray(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(
        create_tray(width=0.25, length=0.25, height=0.05, color=pbu.RED), table
    )

    obj1 = place_object(create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=np.pi / 4))
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def stow_cubby(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(
        create_cubby(width=0.25, length=0.25, height=0.25),
        table,
        x=0.15,
        y=0.3,
        yaw=np.pi,
    )

    obj1 = place_object(create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=np.pi / 4))
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def dump_bin(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(create_bin(width=0.3, length=0.3, height=0.15), table)

    obj1 = place_object(create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=np.pi / 4))
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def dump_bin_color(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    red_region = place_surface(
        create_bin(width=0.25, length=0.25, height=0.12, color=pbu.RED), table, x=-0.15
    )
    yellow_region = place_surface(
        create_bin(width=0.25, length=0.25, height=0.12, color=pbu.YELLOW),
        table,
        x=0.15,
    )
    spacing = 0.12
    apple = place_object(
        create_ycb("apple"), table, Pose2D(yaw=np.pi / 4, x=spacing, y=spacing)
    )
    banana = place_object(create_ycb("banana"), table, Pose2D(yaw=np.pi / 4))
    lemon = place_object(
        create_ycb("lemon"), table, Pose2D(yaw=np.pi / 4, x=-spacing, y=spacing)
    )
    sugar_box = place_object(
        create_ycb("sugar_box"), table, Pose2D(yaw=np.pi / 4, x=spacing, y=-spacing)
    )
    power_drill = place_object(
        create_ycb("mug", use_concave=True),
        table,
        Pose2D(yaw=np.pi / 4, x=-spacing, y=-spacing),
    )

    real_world = create_world(
        robot,
        movable=[apple, banana, lemon, sugar_box, power_drill],
        fixed=obstacles,
        surfaces=[table, red_region, yellow_region],
    )

    return real_world


def stow_cubbies(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(
        create_cubbies(
            get_grid_cells(rows=2, columns=3), width=0.25, length=0.25, height=0.25
        ),
        table,
        x=0.15,
        y=0.25,
        yaw=np.pi,
    )

    obj1 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=np.pi / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def stack_mem(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)

    obj1 = place_object(create_ycb("pudding_box", **kwargs), table, Pose2D(), **kwargs)
    obj2 = place_object(
        create_ycb("gelatin_box", **kwargs), table, Pose2D(y=0.2), **kwargs
    )
    obj3 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(y=-0.1), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1, obj2, obj3], fixed=obstacles, surfaces=[table]
    )

    return real_world


def rearrange_mem(args, robot, **kwargs):
    arms = args.arms
    robot, table, obstacles = create_default_env(robot, robot, arms=arms, **kwargs)

    color_val = pbu.BLUE
    base_color_val = pbu.RED

    obj2 = create_pillar(
        width=0.08, length=0.13, height=0.03, color=color_val, mass=0.15, **kwargs
    )
    place_object_rotate(obj2.body, table, **kwargs)
    obj1 = create_pillar(
        width=0.04, length=0.04, height=0.1, color=color_val, mass=0.1, **kwargs
    )
    place_object_rotate(obj1.body, table, x=0.03, **kwargs)
    obj3 = create_pillar(
        width=0.05, length=0.2, height=0.03, color=base_color_val, mass=0.2, **kwargs
    )
    place_object_rotate(obj3.body, table, y=0.3, **kwargs)

    region = place_surface(create_pillar(**kwargs), table, x=-0.1)
    obj4 = place_object(
        create_ycb("cracker_box", **kwargs), table, Pose2D(x=-0.1), **kwargs
    )

    real_world = create_world(
        robot,
        movable=[obj1, obj2, obj3, obj4],
        fixed=obstacles,
        surfaces=[table, region],
        **kwargs
    )
    return real_world


def stack(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)

    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(), **kwargs
    )
    obj2 = place_object(
        create_ycb("pudding_box", **kwargs), table, Pose2D(y=0.2), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table]
    )
    return real_world


def inspect(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)

    obj1 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=np.pi / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table], **kwargs
    )

    return real_world


def push(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)

    obj1 = place_object(
        create_pillar(
            width=0.19,
            length=0.18,
            height=0.1,
            color=(0.87, 0.72, 0.529, 1),
            mass=0.1,
            **kwargs
        ),
        table,
        **kwargs
    )
    real_world = create_world(robot, movable=[obj1], fixed=obstacles, surfaces=[table])

    return real_world


def pour(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    cup = "teal_cup"  # mug | teal_cup
    bowl = "brown_bowl"  # bowl | brown_bowl
    material = "water"

    obj1 = place_object(
        create_object(cup, use_concave=True), table, Pose2D(yaw=-np.pi / 4), **kwargs
    )
    obj2 = place_object(
        create_object(bowl, **kwargs), table, Pose2D(yaw=np.pi / 4, y=0.3), **kwargs
    )
    real_world = create_world(
        robot,
        movable=[obj1, obj2],
        fixed=obstacles,
        surfaces=[table],
        materials={obj1.body: material},
    )
    assert not args.convex  # need to use concave

    return real_world


def large_obstructing(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(create_pillar(), table, y=0.0)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=np.pi / 4, y=0.3), **kwargs
    )
    obj2 = place_object(
        create_pillar(
            width=0.16,
            length=0.16,
            height=0.1,
            color=(0.87, 0.72, 0.529, 1),
            mass=0.1,
            **kwargs
        ),
        region,
    )
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table, region], **kwargs
    )

    return real_world


def drop_in_bowl(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("strawberry", **kwargs), table, Pose2D(yaw=np.pi / 4), **kwargs
    )
    obj2 = place_object(
        create_ycb("bowl", use_concave=True, **kwargs),
        table,
        Pose2D(yaw=np.pi / 4, y=0.2),
        **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table]
    )

    return real_world


def drop_in_bowl_on_shelf(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)
    region = place_surface(create_pillar(), table, y=0.4)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("strawberry", **kwargs), table, Pose2D(yaw=np.pi / 4, y=-0.05)
    )
    obj2 = place_object(
        create_ycb("bowl", use_concave=True),
        table,
        Pose2D(yaw=np.pi / 4, y=0.16),
        **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


##############################################


def place_object_rotate(
    obj, surface, x=0.0, y=0.0, roll=0.0, pitch=0.0, yaw=0, **kwargs
):
    surface_oobb = surface.get_shape_oobb()
    z = pbu.stable_z_on_aabb(obj, surface_oobb.aabb)  # + Z_EPSILON

    pose_rotate = pbu.Pose(pbu.Point(), pbu.Euler(roll, pitch, yaw))
    pbu.set_pose(obj, pose_rotate)
    _, _, min_z = pbu.get_aabb(obj)[0]  # TODO aabb larger than actual value

    pose = pbu.multiply(
        surface_oobb.pose, pbu.Pose(pbu.Point(x, y, z - min_z)), pose_rotate
    )
    pbu.set_pose(obj, pose)
    return obj


def tight_pack(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)

    region_color_val = pbu.apply_alpha(pbu.BLUE, alpha=0.1)
    region = place_surface(
        create_pillar(width=0.06, length=0.06, color=region_color_val), table, y=0.4
    )
    region2 = place_surface(
        create_pillar(width=0.18, length=0.08, color=region_color_val), table, y=0.2
    )
    real_world = create_world(robot, movable=[], fixed=obstacles, surfaces=[])

    obj1 = create_pillar(width=0.05, length=0.05, height=0.15, color=pbu.BLUE, mass=0.1)
    place_object_rotate(obj1.body, table, roll=np.pi / 2, **kwargs)
    obj2 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(y=0.2, yaw=np.pi / 4), **kwargs
    )

    return real_world


def tight_pack_occlusion(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(robot, arms=arms, **kwargs)

    region_color_val = pbu.apply_alpha(pbu.RED, alpha=0.1)
    region = place_surface(
        create_pillar(width=0.3, length=0.08, color=region_color_val), table, y=0.4
    )
    region2 = place_surface(
        create_pillar(width=0.2, length=0.08, color=region_color_val), table, y=0.2
    )

    real_world = create_world(robot, movable=[], fixed=obstacles, surfaces=[])

    large_obj = "cracker_box"
    obstacle = "mustard_bottle"

    obj1 = create_pillar(width=0.05, length=0.05, height=0.15, color=pbu.RED, mass=0.1)
    place_object_rotate(obj1.body, table, roll=np.pi / 2, **kwargs)
    obj2 = place_object(create_ycb(large_obj), table, Pose2D(x=-0.06), **kwargs)
    obj3 = place_object(create_ycb(obstacle), table, Pose2D(y=0.4), **kwargs)

    return real_world


##############################################


def any_pairwise_collisions(object_list):
    for obj1 in object_list:
        for obj2 in object_list:
            if (obj1 != obj2) and pbu.pairwise_collisions(obj1, [obj2]):
                return True
    return False


def pose_vec(objects):
    return list(itertools.chain(*[pbu.get_pose(obj)[0] for obj in objects]))


def simulate_until_stable(robot, objects, epsilon=1e-4, steps=10, **kwargs):
    obj_poses = pose_vec(objects)
    pbu.enable_gravity()
    for _ in range(steps):
        p.stepSimulation()
    print(np.linalg.norm(np.array(obj_poses) - np.array(pose_vec(objects))))
    while np.linalg.norm(np.array(obj_poses) - np.array(pose_vec(objects))) > epsilon:
        obj_poses = pose_vec(objects)
        for _ in range(steps):
            p.stepSimulation()
    pbu.disable_gravity()


def pose_pile_distance_check(objects, max_distance=0.1, origin=[0.0, 0.0], **kwargs):
    for obj in objects:
        td_pose = pbu.get_pose(obj)[0][:2]
        print(
            "obj: "
            + str(obj)
            + " distance: "
            + str(np.linalg.norm(np.array(td_pose) - np.array(origin)))
        )
        if np.linalg.norm(np.array(td_pose) - np.array(origin)) > max_distance:
            return False
    return True


def sample_object_placements(
    robot, table, picked_objects, td_placement_points, **kwargs
):
    while True:
        z_range = [0.7, 1.0]
        for idx in range(len(picked_objects)):
            x, y = td_placement_points[idx]
            z = random.uniform(*z_range)
            yaw = random.uniform(0, 2 * math.pi)
            pitch = random.uniform(0, 2 * math.pi)
            roll = random.uniform(0, 2 * math.pi)
            pbu.set_pose(
                picked_objects[idx],
                pbu.Pose(
                    pbu.Point(x=x, y=y, z=z), pbu.Euler(yaw=yaw, pitch=pitch, roll=roll)
                ),
            )
        if not any_pairwise_collisions(picked_objects + [table]):
            simulate_until_stable(robot, picked_objects)
            if pose_pile_distance_check(picked_objects):
                break


def pile_of_objects(args, robot, **kwargs):
    arms = args.arms

    seed = 0
    random.seed(seed)
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(create_pillar(width=0.4, length=0.4), table, y=0.4)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    object_name_set = [
        ("potted_meat_can", False),
        ("strawberry", False),
        ("banana", False),
        ("sugar_box", False),
        ("mustard_bottle", False),
    ]
    picked_object_name_set = random.sample(set(object_name_set), 5)
    td_placement_points = [
        (0, 0),
        (-0.05, 0.05),
        (0.05, -0.05),
        (-0.05, -0.05),
        (0.05, 0.05),
    ]

    picked_objects = [
        create_ycb(name, use_concave=concave)
        for name, concave in picked_object_name_set
    ]
    sample_object_placements(robot, table, picked_objects, td_placement_points)

    real_world = create_world(
        robot, movable=picked_objects, fixed=obstacles, surfaces=[table, region]
    )

    return real_world


WORLDS = [
    problem0,
    problem1,
    problem1_color,
    problem1_non_convex,
    problem2,
    problem3,
    place_tray,
    stow_cubby,
    dump_bin,
    stow_cubbies,
    stack,
    inspect,
    push,
    pour,
    large_obstructing,
    drop_in_bowl,
    drop_in_bowl_on_shelf,
    full_occlusion,
    dump_bin_color,
    pile_of_objects,
    tight_pack,
    tight_pack_occlusion,
    problem_five_blocks,
]


def pr2_world_from_problem(problem, robot, args, **kwargs):
    worlds_dict = {fn.__name__: fn for fn in WORLDS}
    return worlds_dict[problem](args, robot, **kwargs)
