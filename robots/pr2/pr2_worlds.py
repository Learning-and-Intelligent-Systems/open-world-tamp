import os

import numpy as np
import pybullet as p
from pybullet_tools.pr2_utils import ARM_NAMES
from pybullet_tools.utils import (
    COLOR_FROM_NAME,
    GREEN,
    PI,
    TEMP_DIR,
    Euler,
    Point,
    Pose,
    apply_alpha,
    get_aabb,
    multiply,
    set_pose,
    stable_z_on_aabb,
    RED,
    BLUE,
    YELLOW,
    pairwise_collision,
    pairwise_collisions,
    get_pose,
    enable_gravity,
    disable_gravity,
    set_camera_pose,
    draw_pose,
    add_data_path,
    HideOutput,
    set_dynamics
)
import random

from open_world.simulation.entities import RealWorld
from open_world.simulation.environment import (
    Pose2D,
    create_pillar,
    create_ycb,
    place_object,
    place_surface,
    create_table_object,
    create_floor_object
)

def create_default_env(**kwargs):

    set_camera_pose(
        camera_point=[0.75, -0.75, 1.25], target_point=[-0.75, 0.75, 0.0], **kwargs
    )
    draw_pose(Pose(), length=1, **kwargs)

    add_data_path()
    with HideOutput(enable=True):
        create_floor_object(**kwargs)
        table = create_table_object(**kwargs)
        obstacles = [table]

        for obst in obstacles:
            set_dynamics(
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
    table, obstacles = create_default_env(**kwargs)
    region = place_surface(
        create_pillar(width=0.3, length=0.3, color=GREEN, **kwargs),
        table,
        yaw=PI / 4,
        **kwargs
    )

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=PI / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region], **kwargs
    )

    return real_world

def create_pybullet_block(color,
                          half_extents, 
                          mass,
                          friction, 
                          orientation):
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    pose = (0, 0, 0)

    # Create the collision shape.
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=half_extents)

    # Create the visual_shape.
    visual_id = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=half_extents,
                                    rgbaColor=color)

    # Create the body.
    block_id = p.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=pose,
                                 baseOrientation=orientation)
    p.changeDynamics(
        block_id,
        linkIndex=-1,  # -1 for the base
        lateralFriction=friction)

    return block_id

def problem_five_blocks(args, robot, **kwargs):
    arms = args.arms  # ARM_NAMES
    table, obstacles = create_default_env(arms=arms, **kwargs)


    table_width=0.6
    table_length=1.2
    table_height=0.73
    table_pose = get_pose(table)[0]

    block_size = 0.045
    _obj_colors = [
        (0.95, 0.05, 0.1, 1.),
        (0.05, 0.95, 0.1, 1.),
        (0.1, 0.05, 0.95, 1.),
        (0.95, 0.95, 0.1, 1.),
        (0.1, 0.1, 0.1, 1.),
    ]

    # Object parameters.
    _obj_mass = 0.5
    _obj_friction = 1.2
    _default_orn = [0.0, 0.0, 0.0, 1.0]

    objs = []
    for i in range(5):
        color = _obj_colors[i % len(_obj_colors)]
        half_extents = (block_size / 2.0, block_size / 2.0,
                        block_size / 2.0)
        objs.append(
            create_pybullet_block(color, half_extents, _obj_mass,
                                    _obj_friction, _default_orn))

    for block_index, block_id in enumerate(objs):
        found_collision_free = False
        timeout = 100
        padding = 0.1
        while(not found_collision_free or timeout>0):
            timeout -= 1
            rx = random.uniform(table_pose[0]-table_width/2.0+padding, table_pose[0]+table_width/2.0-padding)
            ry = random.uniform(table_pose[1]-table_length/2.0+padding, table_pose[1]+table_length/2.0-padding)
            print(rx, ry)
            p.resetBasePositionAndOrientation(
                block_id, [rx, ry, table_height+block_size/2.0],
                _default_orn)
            collision = False
            for placed_block in objs[:block_index]:
                if(pairwise_collision(block_id, placed_block)):
                    collision = True
                    break
            if(not collision):
                break


    real_world = create_world(
        robot, movable=[objs], fixed=obstacles, surfaces=[table], **kwargs
    )

    return real_world


def problem1(args, robot, **kwargs):
    arms = args.arms  # ARM_NAMES
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(
        create_pillar(width=0.3, length=0.3, color=GREEN, **kwargs),
        table,
        yaw=PI / 4,
        **kwargs
    )

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=PI / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region], **kwargs
    )

    return real_world


def from_lisdf(args, robot, **kwargs):
    from lisdf.parsing.parse_sdf import load_sdf

    scene_dir = "/Users/aidancurtis/lisdf/models/m0m"
    sdf_struct = load_sdf(scene_dir)
    world = sdf_struct.aggregate_order[0]
    objects = []
    for mi, model in enumerate(world.models):
        intermediate_path = os.path.join(TEMP_DIR, "{}.sdf".format(str(mi)))
        print(intermediate_path)
        with open(intermediate_path, "w") as f:
            xml_string = "\n".join(model.to_xml_string().split("\n")[1:])
            f.write("<sdf>\n{}\n</sdf>".format(xml_string))
        objects.append(p.loadSDF(intermediate_path))

    arms = args.arms

    real_world = create_world(robot, movable=[], fixed=[], surfaces=[])
    return real_world


def full_occlusion_mem(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(
        create_pillar(width=0.18, color=GREEN, **kwargs), table, **kwargs
    )

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can"), table, Pose2D(yaw=PI / 2, x=0.07), **kwargs
    )
    obj2 = place_object(
        create_ycb("cracker_box", **kwargs), table, Pose2D(x=0), **kwargs
    )
    obj3 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=PI / 4, y=0.4), **kwargs
    )
    place_object_rotate(
        obj3, table, x=0.08, y=0.4, yaw=PI / 2, roll=np.pi / 2, **kwargs
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
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(create_pillar(color=GREEN, **kwargs), table, **kwargs)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can"), table, Pose2D(yaw=PI / 2, x=0.07)
    )
    obj2 = place_object(create_ycb("cracker_box"), table, Pose2D(x=0))
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def problem1_non_convex(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(create_pillar(color=GREEN, **kwargs), table, **kwargs)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can", use_concave=True, **kwargs),
        table,
        Pose2D(yaw=PI / 4),
        **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region], **kwargs
    )
    assert not args.convex

    return real_world


def problem1_color(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(create_pillar(color=GREEN, **kwargs), table, **kwargs)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=PI / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def problem2(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(
        create_pillar(width=0.15, length=0.15, color=GREEN, **kwargs), table, **kwargs
    )

    obj1 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=PI / 4), **kwargs
    )
    obj2 = place_object(create_ycb("tomato_soup_can"), region, Pose2D(), **kwargs)
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table, region]
    )
    return real_world


def problem3(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    red_region = place_surface(
        create_pillar(width=0.15, length=0.15, height=0.05, color=RED), table, y=0.2
    )
    blue_region = place_surface(
        create_pillar(width=0.15, length=0.15, height=0.05, color=BLUE), table, y=0.4
    )

    obj1 = place_object(create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=PI / 4))
    obj2 = place_object(
        create_ycb("tomato_soup_can"), table, Pose2D(y=-0.2, yaw=PI / 8)
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
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(
        create_tray(width=0.25, length=0.25, height=0.05, color=RED), table
    )

    obj1 = place_object(create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=PI / 4))
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def stow_cubby(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(
        create_cubby(width=0.25, length=0.25, height=0.25), table, x=0.15, y=0.3, yaw=PI
    )

    obj1 = place_object(create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=PI / 4))
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def dump_bin(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(create_bin(width=0.3, length=0.3, height=0.15), table)

    obj1 = place_object(create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=PI / 4))
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def dump_bin_color(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    red_region = place_surface(
        create_bin(width=0.25, length=0.25, height=0.12, color=RED), table, x=-0.15
    )
    yellow_region = place_surface(
        create_bin(width=0.25, length=0.25, height=0.12, color=YELLOW), table, x=0.15
    )
    spacing = 0.12
    apple = place_object(
        create_ycb("apple"), table, Pose2D(yaw=PI / 4, x=spacing, y=spacing)
    )
    banana = place_object(create_ycb("banana"), table, Pose2D(yaw=PI / 4))
    lemon = place_object(
        create_ycb("lemon"), table, Pose2D(yaw=PI / 4, x=-spacing, y=spacing)
    )
    sugar_box = place_object(
        create_ycb("sugar_box"), table, Pose2D(yaw=PI / 4, x=spacing, y=-spacing)
    )
    power_drill = place_object(
        create_ycb("mug", use_concave=True),
        table,
        Pose2D(yaw=PI / 4, x=-spacing, y=-spacing),
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
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(
        create_cubbies(
            get_grid_cells(rows=2, columns=3), width=0.25, length=0.25, height=0.25
        ),
        table,
        x=0.15,
        y=0.25,
        yaw=PI,
    )

    obj1 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=PI / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region]
    )

    return real_world


def stack_mem(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)

    obj1 = place_object(create_ycb("pudding_box", **kwargs), table, Pose2D(), **kwargs)
    obj2 = place_object(
        create_ycb("gelatin_box", **kwargs), table, Pose2D(y=0.2), **kwargs
    )
    obj3 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(y=-0.1), **kwargs
    )  # table, Pose2D(y=0.2))
    # place_object_rotate(obj1.body, table, roll=np.pi/2)
    real_world = create_world(
        robot, movable=[obj1, obj2, obj3], fixed=obstacles, surfaces=[table]
    )

    return real_world


def rearrange_mem(args, robot, **kwargs):
    arms = args.arms
    robot, table, obstacles = create_default_env(arms=arms, **kwargs)

    color = "blue"
    color_val = COLOR_FROM_NAME[color]

    base_color = "red"
    base_color_val = COLOR_FROM_NAME[base_color]

    obj2 = create_pillar(
        width=0.08, length=0.13, height=0.03, color=color_val, mass=0.15, **kwargs
    )  # TODO create box
    place_object_rotate(obj2.body, table, **kwargs)
    obj1 = create_pillar(
        width=0.04, length=0.04, height=0.1, color=color_val, mass=0.1, **kwargs
    )  # TODO create box
    place_object_rotate(obj1.body, table, x=0.03, **kwargs)
    obj3 = create_pillar(
        width=0.05, length=0.2, height=0.03, color=base_color_val, mass=0.2, **kwargs
    )  # TODO create box
    place_object_rotate(obj3.body, table, y=0.3, **kwargs)

    region = place_surface(create_pillar(**kwargs), table, x=-0.1)
    obj4 = place_object(
        create_ycb("cracker_box", **kwargs), table, Pose2D(x=-0.1), **kwargs
    )

    # # obj1 = place_object(create_ycb('pudding_box'), table, Pose2D())
    # # obj2 = place_object(create_ycb('gelatin_box'), table, Pose2D(y=0.2))
    # # obj3 = place_object(create_ycb('potted_meat_can'), table, Pose2D(y=-.1)) #table, Pose2D(y=0.2))
    # # # place_object_rotate(obj1.body, table, roll=np.pi/2)
    # # # region = place_surface(create_pillar(width=0.18), table)
    # real_world = create_world(robot, movable=[obj1, obj2, obj3], fixed=obstacles, surfaces=[table])
    real_world = create_world(
        robot,
        movable=[obj1, obj2, obj3, obj4],
        fixed=obstacles,
        surfaces=[table, region],
        **kwargs
    )

    # TODO: describe using category
    return real_world


def stack(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)

    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(), **kwargs
    )
    obj2 = place_object(
        create_ycb("pudding_box", **kwargs), table, Pose2D(y=0.2), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table]
    )

    # NOTE need to turn on '-o' flag
    return real_world


def inspect(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)

    obj1 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=PI / 4), **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[table], **kwargs
    )

    # TODO: describe using category
    return real_world


def push(args, robot, **kwargs):
    arms = args.arms
    # arms = ARM_NAMES
    table, obstacles = create_default_env(arms=arms, **kwargs)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
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
    # arms = ARM_NAMES
    table, obstacles = create_default_env(arms=arms, **kwargs)
    cup = "teal_cup"  # mug | teal_cup
    bowl = "brown_bowl"  # bowl | brown_bowl
    material = "water"

    obj1 = place_object(
        create_object(cup, use_concave=True), table, Pose2D(yaw=-PI / 4), **kwargs
    )
    obj2 = place_object(
        create_object(bowl, **kwargs), table, Pose2D(yaw=PI / 4, y=0.3), **kwargs
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
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(create_pillar(), table, y=0.0)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(yaw=PI / 4, y=0.3), **kwargs
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

    # assume: no assumption on observation. if not detected then declare success
    return real_world


def drop_in_bowl(args, robot, **kwargs):
    arms = args.arms
    # arms = ARM_NAMES
    table, obstacles = create_default_env(arms=arms, **kwargs)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("strawberry", **kwargs), table, Pose2D(yaw=PI / 4), **kwargs
    )
    obj2 = place_object(
        create_ycb("bowl", use_concave=True, **kwargs),
        table,
        Pose2D(yaw=PI / 4, y=0.2),
        **kwargs
    )
    real_world = create_world(
        robot, movable=[obj1, obj2], fixed=obstacles, surfaces=[table]
    )

    return real_world


def drop_in_bowl_on_shelf(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)
    region = place_surface(create_pillar(), table, y=0.4)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = place_object(
        create_ycb("strawberry", **kwargs), table, Pose2D(yaw=PI / 4, y=-0.05)
    )
    obj2 = place_object(
        create_ycb("bowl", use_concave=True),
        table,
        Pose2D(yaw=PI / 4, y=0.16),
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
    z = stable_z_on_aabb(obj, surface_oobb.aabb)  # + Z_EPSILON

    pose_rotate = Pose(Point(), Euler(roll, pitch, yaw))
    set_pose(obj, pose_rotate)
    _, _, min_z = get_aabb(obj)[0]  # TODO aabb larger than actual value

    pose = multiply(surface_oobb.pose, Pose(Point(x, y, z - min_z)), pose_rotate)
    set_pose(obj, pose)
    return obj


def tight_pack_mem(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)

    region_color = "red"
    region_color_val = COLOR_FROM_NAME[region_color]
    region = place_surface(
        create_pillar(width=0.08, length=0.22, color=region_color_val), table, x=0.15
    )  # , y=0.4)
    real_world = create_world(
        robot, movable=[], fixed=obstacles, surfaces=[table, region, region2]
    )

    color = "blue"
    color_val = COLOR_FROM_NAME[color]

    obj1 = create_pillar(
        width=0.05, length=0.05, height=0.05, color=color_val, mass=0.1
    )  # TODO create box
    place_object_rotate(obj1.body, table, roll=np.pi / 2, **kwargs)
    obj2 = place_object(
        create_ycb(args.ycb, **kwargs, **kwargs),
        table,
        Pose2D(y=0.2, yaw=np.pi / 4),
        **kwargs
    )

    return real_world


def tight_pack(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)

    region_color = "red"
    region_color_val = apply_alpha(COLOR_FROM_NAME[region_color], alpha=0.1)
    region = place_surface(
        create_pillar(width=0.06, length=0.06, color=region_color_val), table, y=0.4
    )
    region2 = place_surface(
        create_pillar(width=0.18, length=0.08, color=region_color_val), table, y=0.2
    )
    real_world = create_world(robot, movable=[], fixed=obstacles, surfaces=[])

    # cup = "simplified_teal_cup"
    color = "blue"
    color_val = COLOR_FROM_NAME[color]

    obj1 = create_pillar(
        width=0.05, length=0.05, height=0.15, color=color_val, mass=0.1
    )  # TODO create box
    place_object_rotate(obj1.body, table, roll=np.pi / 2, **kwargs)
    obj2 = place_object(
        create_ycb(args.ycb, **kwargs), table, Pose2D(y=0.2, yaw=np.pi / 4), **kwargs
    )

    return real_world


def tight_pack_occlusion(args, robot, **kwargs):
    arms = args.arms
    table, obstacles = create_default_env(arms=arms, **kwargs)

    region_color = "red"
    color = "red"
    color_val = COLOR_FROM_NAME[color]

    region_color_val = apply_alpha(COLOR_FROM_NAME[region_color], alpha=0.1)
    region = place_surface(
        create_pillar(width=0.3, length=0.08, color=region_color_val), table, y=0.4
    )
    region2 = place_surface(
        create_pillar(width=0.2, length=0.08, color=region_color_val), table, y=0.2
    )

    real_world = create_world(robot, movable=[], fixed=obstacles, surfaces=[])

    large_obj = "cracker_box"
    obstacle = "mustard_bottle"

    obj1 = create_pillar(
        width=0.05, length=0.05, height=0.15, color=color_val, mass=0.1
    )  # TODO create box
    place_object_rotate(obj1.body, table, roll=np.pi / 2, **kwargs)
    obj2 = place_object(create_ycb(large_obj), table, Pose2D(x=-0.06), **kwargs)
    obj3 = place_object(create_ycb(obstacle), table, Pose2D(y=0.4), **kwargs)

    return real_world


##############################################


def any_pairwise_collisions(object_list):
    for obj1 in object_list:
        for obj2 in object_list:
            if (obj1 != obj2) and pairwise_collisions(obj1, [obj2]):
                return True
    return False


def pose_vec(objects):
    return list(itertools.chain(*[get_pose(obj)[0] for obj in objects]))


def simulate_until_stable(robot, objects, epsilon=1e-4, steps=10, **kwargs):
    obj_poses = pose_vec(objects)
    enable_gravity()
    for _ in range(steps):
        p.stepSimulation()
    print(np.linalg.norm(np.array(obj_poses) - np.array(pose_vec(objects))))
    while np.linalg.norm(np.array(obj_poses) - np.array(pose_vec(objects))) > epsilon:
        obj_poses = pose_vec(objects)
        for _ in range(steps):
            p.stepSimulation()
    disable_gravity()
    set_default_conf(robot)


def pose_pile_distance_check(objects, max_distance=0.1, origin=[0.0, 0.0], **kwargs):
    for obj in objects:
        td_pose = get_pose(obj)[0][:2]
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
            set_pose(
                picked_objects[idx],
                Pose(Point(x=x, y=y, z=z), Euler(yaw=yaw, pitch=pitch, roll=roll)),
            )
        if not any_pairwise_collisions(picked_objects + [table]):
            simulate_until_stable(robot, picked_objects)
            if pose_pile_distance_check(picked_objects):
                break


def pile_of_objects(args, robot, **kwargs):
    arms = args.arms

    seed = 0
    random.seed(seed)
    # arms = ARM_NAMES
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

    # obj2 = place_object(create_ycb("bowl", use_concave=True), table, Pose2D(yaw=PI / 4, y=0.16))

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
    from_lisdf,
    problem_five_blocks
]


def pr2_world_from_problem(problem, robot, args, **kwargs):
    worlds_dict = {fn.__name__: fn for fn in WORLDS}
    return worlds_dict[problem](args, robot, **kwargs)
