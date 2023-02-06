from pybullet_tools.utils import (
    GREEN,
    PI,
    RGBA,
    TAN,
    HideOutput,
    Point,
    Pose,
    add_data_path,
    create_box,
    load_model,
    draw_pose,
    joint_from_name,
    set_joint_position,
    set_camera_pose,
    set_dynamics,
    set_pose,
    create_box,
    AABB,
    Euler,
    get_link_names,
    get_all_links
)

import random

from open_world.simulation.entities import Object, RealWorld
from open_world.simulation.environment import (
    Pose2D,
    create_floor_object,
    create_pillar,
    create_table_object,
    create_ycb,
    place_object,
    place_surface,
)
from open_world.exploration.utils import Room, LIGHT_GREY, GRID_HEIGHT
import math
import pybullet as p


def create_world(robot, movable=[], attachable=[], fixed=[], surfaces=[], room=None, **kwargs):
    obstacles = sorted(set(fixed) | set(surfaces))
    return RealWorld(
        robot,
        movable=movable,
        attachable=attachable,
        fixed=obstacles,
        detectable=movable,
        known=obstacles,
        surfaces=surfaces,
        room=room,
        **kwargs
    )


#######################################################


def create_default_env(**kwargs):
    # TODO: p.loadSoftBody
    set_camera_pose(
        camera_point=[0.75, -0.75, 1.25], target_point=[-0.75, 0.75, 0.0], **kwargs
    )
    draw_pose(Pose(), length=1, **kwargs)

    add_data_path()
    with HideOutput(enable=True):
        floor = create_floor_object(**kwargs)
        table = create_table_object(**kwargs)
        set_pose(table, Pose([1.0, 0, 0]), **kwargs)
        obstacles = [
            # floor, # collides with the robot when MAX_DISTANCE >= 5e-3
            table,
        ]

        for obst in obstacles:
            # print(get_dynamics_info(obst))
            set_dynamics(
                obst,
                lateralFriction=1.0,  # linear (lateral) friction
                spinningFriction=1.0,  # torsional friction around the contact normal
                rollingFriction=0.01,  # torsional friction orthogonal to contact normal
                restitution=0.0,  # restitution: 0 => inelastic collision, 1 => elastic collision
                **kwargs
            )

    return table, obstacles


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

def red_block_mobile(args, robot, vg=None, **kwargs):
    floor_size = 6
    floor = create_pillar(width=floor_size, length=floor_size, color=TAN, **kwargs)
    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    side = 0.05
    box_mass = 0.2
    height = side * 15
    red_box = Object(
        create_box(
            w=side,
            l=side,
            h=height,
            color=RGBA(219 / 256.0, 50 / 256.0, 54 / 256.0, 1.0),
            mass=box_mass,
            **kwargs
        ), **kwargs
    )

    if(vg != None):
        x, y = random.choice(vg.get_frontier())
        block_pose = Pose(point=Point(x = x-(floor_size / 2), y = y-(floor_size / 2), z = height / 2.0))
    else:
        block_pose = Pose(point=Point(x=random.uniform(-floor_size / 2, floor_size / 2), y=random.uniform(-floor_size / 2, floor_size / 2), z = height / 2.0)) 


    set_pose(red_box, block_pose, **kwargs)
    real_world = create_world(robot, movable=[red_box], fixed=[], surfaces=[floor], **kwargs)
    return real_world


def vanamo_m0m_chair(args, robot, **kwargs):
    return vanamo_m0m(args, robot, has_blocking_chair=True, **kwargs)

def vanamo_m0m(args, robot, has_blocking_chair=False, **kwargs):
    width = 4
    length = 6
    wall_height = 2
    center = [1, 2]

    floor1 = create_pillar(width=width, length=length, color=TAN, **kwargs)
    set_pose(floor1, Pose(Point(x=center[0], y=center[1])), **kwargs)

    wall_thickness = 0.1
    wall_1 = create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY, **kwargs)
    set_pose(wall_1,
                Pose(point=Point(x=center[0], y=center[1] + length / 2 + wall_thickness / 2, z=wall_height / 2)), **kwargs)

    wall_2 = create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY, **kwargs)
    set_pose(wall_2,
                Pose(point=Point(x=center[0], y=center[1] - (length / 2 + wall_thickness / 2), z=wall_height / 2)), **kwargs)

    wall_3 = create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY, **kwargs)
    set_pose(wall_3,
                Pose(point=Point(y=center[1], x=center[0] + width / 2 + wall_thickness / 2, z=wall_height / 2)), **kwargs)

    wall_4 = create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY, **kwargs)
    set_pose(wall_4,
                Pose(point=Point(y=center[1], x=center[0] - (width / 2 + wall_thickness / 2), z=wall_height / 2)), **kwargs)

    wall_5 = create_pillar(length=4.4, width=wall_thickness, height=wall_height, color=LIGHT_GREY, **kwargs)
    set_pose(wall_5,
                Pose(point=Point(y=1.2, x=1.2, z=wall_height / 2)), **kwargs)

    walls = [wall_1, wall_2, wall_3, wall_4, wall_5]
    floors = [floor1]
    aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                upper=(center[0] + width / 2.0, center[1] + length / 2.0, GRID_HEIGHT))

    movable_obstacles = []
    if(has_blocking_chair):
        blocking_chair1 = load_model("../models/partnet_mobility/179/mobility.urdf", scale=0.5, **kwargs)
        chair_color = (0.8,0.8,0,1)
        link_names = get_link_names(blocking_chair1, get_all_links(blocking_chair1, **kwargs), **kwargs)
        set_joint_position(blocking_chair1, 17, math.pi, **kwargs)

        kwargs["client"].changeVisualShape(blocking_chair1, link_names.index("link_15"), rgbaColor=chair_color)
        kwargs["client"].changeVisualShape(blocking_chair1, link_names.index("link_15_helper"), rgbaColor=chair_color)
        
        chair_pos1 = (1.4, 4.3, 0.42)
        set_pose(blocking_chair1, Pose(point=Point(x=chair_pos1[0], y=chair_pos1[1], z=chair_pos1[2])), **kwargs)

        movable_obstacles = [blocking_chair1]
        print("Movable obstacles: "+str(movable_obstacles))

    room = Room(walls, floors, aabb, movable_obstacles)

    add_data_path()
    with HideOutput(enable=True):
        table = create_table_object(color=LIGHT_GREY, **kwargs)
        set_pose(table, Pose(Point(2.2, 0, 0), Euler(yaw=math.pi/2.0)), **kwargs)
        obstacles = [
            table,
        ]

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
        robot, movable=[obj1], fixed=obstacles, surfaces=[table, region], room=room, **kwargs
    )

    return real_world

def namo(args, robot, random_widths=False, **kwargs):
    
    def movability_prior(box_width, mean_width):
        # return np.random.choice([0, 1], p=[0.7, 0.3])
        return int(box_width < mean_width)

    floor_size = 6
    floor = create_pillar(width=floor_size, length=floor_size, color=TAN, **kwargs)
    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    side = 0.05
    box_mass = 0.2
    height = side * 16

    box_x, box_y = -2.5, 0

    table = create_table_object(**kwargs)
    set_pose(table, Pose([box_x, box_y, 0]), **kwargs)
    obj1 = place_object(
        create_ycb("potted_meat_can", **kwargs), table, Pose2D(yaw=PI / 4), **kwargs
    )
   
    # Don't set the pose of the robot, only the joint confs
    robot.set_group_positions("base", [2, 0, math.pi, 0])
   
    total_width = 0
    spacing = 0.65
    block_height = 0.8
    blocking_boxes = []
    mean_box_width = 0.3
    std_box_width = 0.2
    colors = [RGBA(0.3, 0.3, 0.3, 1.0), RGBA(54 / 256.0, 50 / 256.0, 219 / 256.0, 1.0)]
    width_list = []
    if random_widths:
        while total_width < floor_size:
            block_width = random.uniform(
                mean_box_width - std_box_width, mean_box_width + std_box_width
            )
            if total_width + block_width + spacing > floor_size:
                break
            blocking_box = Object(
                create_box(
                    w=block_width,
                    l=block_width,
                    h=block_width,
                    color=colors[
                        movability_prior(block_width, mean_width=mean_box_width)
                    ],
                    mass=box_mass,
                    **kwargs
                )
            )
            set_pose(
                blocking_box,
                Pose(
                    point=Point(
                        x=-block_width / 2,
                        y=-floor_size / 2 + total_width + spacing + block_width / 2,
                        z=block_width / 2.0,
                    )
                ), **kwargs
            )

            total_width += block_width
            total_width += spacing
            blocking_boxes.append(blocking_box)
            width_list.append(block_width)
    else:
        width_list = [
            0.35954404643140534,
            0.2506706557568167,
            0.14700198265249174,
            0.1129144419011968,
            0.179910954976249,
            0.19335698094573505,
            0.1554801454275457   ]
        for block_width in width_list:
            blocking_box = Object(load_model(
                "../models/partnet_mobility/179/mobility.urdf", scale=0.4, **kwargs
            ), **kwargs)
            set_joint_position(blocking_box, 17, random.uniform(-math.pi, math.pi), **kwargs)
            set_pose(
                blocking_box,
                Pose(
                    point=Point(
                        x=-block_width / 2,
                        y=-floor_size / 2 + total_width + spacing/2.0 + block_width / 2,
                        z=block_height / 2.0,
                    )
                ), **kwargs
            )

            total_width += block_width
            total_width += spacing
            blocking_boxes.append(blocking_box)

    BUFFER = 0.5
    robot.custom_limits[joint_from_name(robot, 'x', **kwargs)] = (-floor_size/2.0+BUFFER, floor_size/2.0-BUFFER)
    robot.custom_limits[joint_from_name(robot, 'y', **kwargs)] = (-floor_size/2.0+BUFFER, floor_size/2.0-BUFFER)

    real_world = create_world(
        robot, movable=[obj1], attachable=blocking_boxes, fixed=[], surfaces=[floor], **kwargs
    )
    return real_world

def namo_old(args, robot, **kwargs):

    floor_size = 6
    floor = create_pillar(width=floor_size, length=floor_size, color=TAN, **kwargs)
    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    side = 0.05
    box_mass = 0.2
    height = side * 8
    red_box = Object(
        create_box(
            w=side,
            l=side,
            h=height,
            color=RGBA(219 / 256.0, 50 / 256.0, 54 / 256.0, 1.0),
            mass=box_mass,
            **kwargs
        ), **kwargs
    )

    box_x, box_y = -2, 0

    # Don't set the pose of the robot, only the joint confs
    # set_pose(robot, Pose(point=Point(x=2, y=0), euler=Euler(0, 0, math.pi)))
    robot.set_group_positions("base", [2, 0, math.pi])
    set_pose(red_box, Pose(point=Point(x=box_x, y=box_y, z=height / 2.0)), **kwargs)


    def movability_prior(box_width, mean_width):
        # return np.random.choice([0, 1], p=[0.7, 0.3])
        return int(box_width < mean_width)



    total_width = 0
    spacing = 0.2
    blocking_boxes = []
    mean_box_width = 0.3
    colors = [RGBA(0.3, 0.3, 0.3, 1.0), RGBA(54 / 256.0, 50 / 256.0, 219 / 256.0, 1.0)]  
    width_list = [
        0.45954404643140534,
        0.3506706557568167,
        0.24700198265249174,
        0.2129144419011968,
        0.479910954976249,
        0.49335698094573505,
        0.2554801454275457,
        0.2738256213163567,
        0.21771202876702336,
        0.34203547468926415,
        0.28426359890736475,
    ]
    for block_width in width_list:
        blocking_box = Object(
            create_box(
                w=block_width,
                l=block_width,
                h=block_width,
                color=colors[
                    movability_prior(block_width, mean_width=mean_box_width)
                ],
                mass=box_mass,
                **kwargs
            ), **kwargs
        )
        set_pose(
            blocking_box,
            Pose(
                point=Point(
                    x=-block_width / 2,
                    y=-floor_size / 2 + total_width + spacing + block_width / 2,
                    z=block_width / 2.0,
                )
            ), **kwargs
        )

        total_width += block_width
        total_width += spacing
        blocking_boxes.append(blocking_box)

    real_world = create_world(
        robot, movable=[red_box] + blocking_boxes, fixed=[], surfaces=[floor], **kwargs
    )
    return real_world

def empty(args, robot, **kwargs):
    floor = create_pillar(width=6, length=6, color=TAN, **kwargs)
    real_world = create_world(robot, movable=[], fixed=[], surfaces=[floor], **kwargs)
    return real_world


WORLDS = [problem0, empty, red_block_mobile, namo, vanamo_m0m, vanamo_m0m_chair]


def movo_world_from_problem(problem, robot, args, **kwargs):
    worlds_dict = {fn.__name__: fn for fn in WORLDS}
    return worlds_dict[problem](args, robot, **kwargs)
