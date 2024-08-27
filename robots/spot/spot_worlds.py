import random

from open_world.exploration.utils import GRID_HEIGHT, LIGHT_GREY, Room
from open_world.simulation.entities import RealWorld
from open_world.simulation.environment import (Pose2D, create_floor_object,
                                               create_pillar,
                                               create_table_object, create_ycb,
                                               place_object, place_surface)
from pybullet_tools.utils import (AABB, GREEN, PI, TAN, HideOutput, Point,
                                  Pose, add_data_path, draw_pose,
                                  set_camera_pose, set_dynamics, set_pose)


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
    # TODO: p.loadSoftBody
    set_camera_pose(
        camera_point=[0.75, -0.75, 1.25], target_point=[-0.75, 0.75, 0.0], **kwargs
    )
    draw_pose(Pose(), length=1, **kwargs)

    add_data_path()
    with HideOutput(enable=True):
        floor = create_floor_object(**kwargs)
        table = create_table_object(height=0.6, **kwargs)
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


def empty_room(args, robot, **kwargs):
    width = 4
    length = 4
    wall_height = 2
    center = [1, 0]

    floor1 = create_pillar(width=width, length=length, color=TAN, **kwargs)
    set_pose(floor1, Pose(Point(x=center[0], y=center[1])), **kwargs)

    wall_thickness = 0.1
    wall_1 = create_pillar(
        width=width,
        length=wall_thickness,
        height=wall_height,
        color=LIGHT_GREY,
        **kwargs
    )
    set_pose(
        wall_1,
        Pose(
            point=Point(
                x=center[0],
                y=center[1] + length / 2 + wall_thickness / 2,
                z=wall_height / 2,
            )
        ),
        **kwargs
    )

    wall_2 = create_pillar(
        width=width,
        length=wall_thickness,
        height=wall_height,
        color=LIGHT_GREY,
        **kwargs
    )
    set_pose(
        wall_2,
        Pose(
            point=Point(
                x=center[0],
                y=center[1] - (length / 2 + wall_thickness / 2),
                z=wall_height / 2,
            )
        ),
        **kwargs
    )

    wall_3 = create_pillar(
        length=length,
        width=wall_thickness,
        height=wall_height,
        color=LIGHT_GREY,
        **kwargs
    )
    set_pose(
        wall_3,
        Pose(
            point=Point(
                y=center[1],
                x=center[0] + width / 2 + wall_thickness / 2,
                z=wall_height / 2,
            )
        ),
        **kwargs
    )

    wall_4 = create_pillar(
        length=length,
        width=wall_thickness,
        height=wall_height,
        color=LIGHT_GREY,
        **kwargs
    )
    set_pose(
        wall_4,
        Pose(
            point=Point(
                y=center[1],
                x=center[0] - (width / 2 + wall_thickness / 2),
                z=wall_height / 2,
            )
        ),
        **kwargs
    )

    walls = [wall_1, wall_2, wall_3, wall_4]

    floors = [floor1]
    aabb = AABB(
        lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
        upper=(center[0] + width / 2.0, center[1] + length / 2.0, GRID_HEIGHT),
    )

    room = Room(walls, floors, aabb, [])

    real_world = create_world(
        robot, movable=[], fixed=[], surfaces=[], room=room, **kwargs
    )

    return real_world


WORLDS = [problem0, empty_room]


def spot_world_from_problem(problem, robot, args, **kwargs):
    worlds_dict = {fn.__name__: fn for fn in WORLDS}
    return worlds_dict[problem](args, robot, **kwargs)
