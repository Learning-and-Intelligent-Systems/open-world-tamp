import math
import random

import numpy as np

import owt.pb_utils as pbu
from owt.simulation.entities import Object, RealWorld
from owt.simulation.environment import (Pose2D, create_floor_object,
                                        create_pillar, create_table_object,
                                        create_ycb, place_object,
                                        place_surface)


def create_world(
    robot, movable=[], attachable=[], fixed=[], surfaces=[], room=None, **kwargs
):
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
    pbu.set_camera_pose(
        camera_point=[0.75, -0.75, 1.25], target_point=[-0.75, 0.75, 0.0], **kwargs
    )
    pbu.draw_pose(pbu.Pose(), length=1, **kwargs)

    pbu.add_data_path()
    with pbu.HideOutput(enable=True):
        floor = create_floor_object(**kwargs)
        table = create_table_object(**kwargs)
        pbu.set_pose(table, pbu.Pose([1.0, 0, 0]), **kwargs)
        obstacles = [
            table,
        ]

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


def problem0(args, robot, **kwargs):
    table, obstacles = create_default_env(**kwargs)
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


def red_block_mobile(args, robot, vg=None, **kwargs):
    floor_size = 6
    floor = create_pillar(width=floor_size, length=floor_size, color=pbu.TAN, **kwargs)
    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    side = 0.05
    box_mass = 0.2
    height = side * 15
    red_box = Object(
        pbu.create_box(
            w=side,
            l=side,
            h=height,
            color=pbu.RGBA(219 / 256.0, 50 / 256.0, 54 / 256.0, 1.0),
            mass=box_mass,
            **kwargs
        ),
        **kwargs
    )

    if vg != None:
        x, y = random.choice(vg.get_frontier())
        block_pose = pbu.Pose(
            point=pbu.Point(
                x=x - (floor_size / 2), y=y - (floor_size / 2), z=height / 2.0
            )
        )
    else:
        block_pose = pbu.Pose(
            point=pbu.Point(
                x=random.uniform(-floor_size / 2, floor_size / 2),
                y=random.uniform(-floor_size / 2, floor_size / 2),
                z=height / 2.0,
            )
        )

    pbu.set_pose(red_box, block_pose, **kwargs)
    real_world = create_world(
        robot, movable=[red_box], fixed=[], surfaces=[floor], **kwargs
    )
    return real_world


def empty(args, robot, **kwargs):
    floor = create_pillar(width=6, length=6, color=pbu.TAN, **kwargs)
    real_world = create_world(robot, movable=[], fixed=[], surfaces=[floor], **kwargs)
    return real_world


WORLDS = [
    problem0,
    empty,
    red_block_mobile,
]


def movo_world_from_problem(problem, robot, args, **kwargs):
    worlds_dict = {fn.__name__: fn for fn in WORLDS}
    return worlds_dict[problem](args, robot, **kwargs)
