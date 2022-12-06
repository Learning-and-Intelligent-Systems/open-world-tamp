from pybullet_tools.utils import (
    GREEN,
    RGBA,
    HideOutput,
    Point,
    Pose,
    add_data_path,
    create_box,
    load_model,
    set_camera_pose,
    set_dynamics,
    set_pose,
)


from open_world.simulation.entities import Object, RealWorld, Shape
from open_world.simulation.environment import (
    create_floor_object,
    create_pillar,
    create_ycb,
    place_surface,
)


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


def create_default_env(client=None, **kwargs):
    # TODO: p.loadSoftBody
    set_camera_pose(
        camera_point=[0.75, -0.75, 1.25], target_point=[-0.75, 0.75, 0.0], client=client
    )
    # draw_pose(Pose(), length=1)

    add_data_path()
    with HideOutput(enable=True):
        floor = create_floor_object(client=client)
        set_pose(floor, Pose(point=Point(z=-0.04)), client=client)
        obstacles = [
            floor,  # collides with the robot when MAX_DISTANCE >= 5e-3
            # table,
        ]

        for obst in obstacles:
            # print(get_dynamics_info(obst))
            set_dynamics(
                obst,
                lateralFriction=1.0,  # linear (lateral) friction
                spinningFriction=1.0,  # torsional friction around the contact normal
                rollingFriction=0.01,  # torsional friction orthogonal to contact normal
                restitution=0.0,  # restitution: 0 => inelastic collision, 1 => elastic collision
                client=client,
            )

    return floor, obstacles


def problem0(args, robot, client=None):
    arms = args.arms  # ARM_NAMES
    floor, obstacles = create_default_env(arms=arms, client=client)
    region = create_pillar(width=0.3, length=0.3, color=GREEN, client=client)
    set_pose(region, Pose(point=Point(x=0.5, y=0, z=-0.02)), client=client)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    obj1 = create_ycb("potted_meat_can", client=client)
    set_pose(obj1, Pose(point=Point(x=0.5, y=-0.5, z=0.03)), client=client)

    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[floor, region], client=client
    )

    return real_world


def problem0_block(args, robot, client=None):
    arms = args.arms  # ARM_NAMES
    floor, obstacles = create_default_env(arms=arms, client=client)
    region = create_pillar(width=0.3, length=0.3, color=GREEN, client=client)
    set_pose(region, Pose(point=Point(x=0.5, y=0, z=-0.02)), client=client)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    box_mass = 0.2
    side = 0.05
    obj1 = Object(
        create_box(
            w=side,
            l=side,
            h=side,
            color=RGBA(219 / 256.0, 50 / 256.0, 54 / 256.0, 1.0),
            mass=box_mass,
        ),
        client=client,
    )
    place_surface(floor, x=0.5, y=-0.5, client=client)

    real_world = create_world(
        robot, movable=[obj1], fixed=obstacles, surfaces=[floor, region], client=client
    )

    return real_world


def problem_cabinet(args, robot, client=None):
    arms = args.arms  # ARM_NAMES
    floor, obstacles = create_default_env(arms=arms, client=client)
    cabinet = load_model(
        "../models/partnet_mobility/46236/mobility.urdf", scale=0.5, client=client
    )
    set_pose(cabinet, Pose(point=Point(x=0.7, z=0.44)), client=client)
    real_world = create_world(
        robot, movable=[], fixed=obstacles, surfaces=[floor], client=client
    )

    return real_world


def block_stack(args, robot, client=None):
    arms = args.arms  # ARM_NAMES
    floor, obstacles = create_default_env(arms=arms, client=client)
    region = create_pillar(width=0.3, length=0.3, color=GREEN, client=client)
    set_pose(region, Pose(point=Point(x=0.5, y=0, z=-0.02)), client=client)

    # cracker_box | tomato_soup_can | potted_meat_can | bowl
    side = 0.05
    box_mass = 0.2
    red_box = Object(
        create_box(
            w=side,
            l=side,
            h=side,
            color=RGBA(219 / 256.0, 50 / 256.0, 54 / 256.0, 1.0),
            mass=box_mass,
            client=client,
        ),
        client=client,
    )
    blue_box = Object(
        create_box(
            w=side,
            l=side,
            h=side,
            color=RGBA(244 / 256.0, 194 / 256.0, 13 / 256.0, 1.0),
            mass=box_mass,
            client=client,
        ),
        client=client,
    )
    yellow_box = Object(
        create_box(
            w=side,
            l=side,
            h=side,
            color=RGBA(72 / 256.0, 133 / 256.0, 237 / 256.0, 1.0),
            mass=box_mass,
            client=client,
        ),
        client=client,
    )
    set_pose(red_box, Pose(point=Point(x=0.505, y=-0.5, z=-0.005)), client=client)
    set_pose(
        blue_box, Pose(point=Point(x=0.495, y=-0.5, z=-0.005 + side)), client=client
    )
    set_pose(
        yellow_box,
        Pose(point=Point(x=0.5, y=-0.505, z=-0.005 + 2 * side)),
        client=client,
    )
    real_world = create_world(
        robot,
        movable=[red_box, blue_box, yellow_box],
        fixed=obstacles,
        surfaces=[floor, region],
        client=client,
    )

    return real_world


WORLDS = [problem0, problem0_block, block_stack, problem_cabinet]


def panda_world_from_problem(problem, robot, args, client=None):
    worlds_dict = {fn.__name__: fn for fn in WORLDS}
    return worlds_dict[problem](args, robot, client=client)
