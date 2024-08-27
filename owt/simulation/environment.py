from __future__ import print_function

import os
import string
import sys
from itertools import product

import numpy as np
import pybullet as p
from open_world.estimation.geometry import project_base_points
from open_world.simulation.entities import Object, Shape
from open_world.simulation.lis import (BOWLS_PATH, CUPS_PATH, YCB_COLORS,
                                       YCB_MASSES, get_ycb_obj_path)
from pybullet_tools.pr2_problems import create_floor, create_table
from pybullet_tools.utils import (BASE_LINK, CLIENT, DEFAULT_CLIENT, GREY,
                                  NULL_ID, PI, STATIC_MASS, TAN, WHITE, Euler,
                                  Point, Pose, aabb_from_points, apply_alpha,
                                  body_collision, convex_combination,
                                  create_body, create_box,
                                  create_collision_shape, create_mesh,
                                  create_obj, create_plane, create_shape_array,
                                  create_visual_shape, get_aabb_center,
                                  get_aabb_extent, get_box_geometry,
                                  get_link_subtree, get_mesh_geometry,
                                  get_pose, load_pybullet, mesh_from_points,
                                  multiply, point_from_pose, pose_from_pose2d,
                                  quat_from_pose, read_obj, set_all_color,
                                  set_dynamics, set_pose, set_texture,
                                  stable_z_on_aabb, unit_point, unit_quat)

# NOTE(caelan): must come before other imports
sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
        #'pddlstream/examples/pybullet/utils',
        #'../ltamp_pr2',
    ]
)
import math

import pybullet_tools

pybullet_tools.utils.TEMP_DIR = "temp_meshes/"  # TODO: resolve conflict with pddlstream

from open_world.simulation.entities import Object

DRAKE_YCB_PATH = "/Users/caelan/Programs/external/" "drake/manipulation/models/ycb/sdf"
PYBULLET_YCB_DIR = (
    "/Users/caelan/Programs/external/pybullet/"
    "pybullet-object-models/pybullet_object_models/ycb_objects"
)


def create_ycb(
    name,
    mesh_type="centered",
    project_base=False,
    z_threshold=1e-2,
    mass_scale=1e-1,
    use_concave=False,
    client=None,
    **kwargs
):
    client = client or DEFAULT_CLIENT
    concave_ycb_path = get_ycb_obj_path(name, use_concave=use_concave)
    ycb_path = get_ycb_obj_path(name)

    full_name = os.path.basename(os.path.dirname(ycb_path))
    mass = mass_scale * YCB_MASSES[name]
    # mass = STATIC_MASS

    # TODO: separate visual and collision boddies
    if (name in YCB_COLORS) and use_concave:
        color = YCB_COLORS[name]
    else:
        color = WHITE

    print(name, mesh_type, full_name)
    if mesh_type == "drake":
        # https://github.com/RobotLocomotion/drake/blob/3b33ed5760d4030f4df04241be3ab6958069e03d/manipulation/models/ycb/sdf/010_potted_meat_can.sdf
        path = os.path.join(DRAKE_YCB_PATH, "{}.sdf".format(full_name))
        reference_pose = Pose(
            euler=Euler(roll=-PI / 2, yaw=-PI / 2)
        )  # The following two are equal
        # reference_pose = multiply(Pose(euler=Euler(yaw=-PI/2)), Pose(euler=Euler(roll=-PI/2)))
        [body] = load_pybullet(path)
        set_pose(body, reference_pose)
    elif mesh_type == "pybullet":
        # https://github.com/eleramp/pybullet-object-models/blob/master/pybullet_object_models/ycb_objects/YcbPottedMeatCan/model.urdf
        path = os.path.join(
            PYBULLET_YCB_DIR,
            "Ycb{}/model.urdf".format(
                "".join(word.capitalize() for word in name.split("_"))
            ),
        )
        body = load_pybullet(path)
        reference_pose = Pose(euler=Euler(yaw=0))
        set_pose(body, reference_pose)
    elif mesh_type == "centered":
        import trimesh

        mesh = trimesh.load(ycb_path)

        # TODO: separate visual and collision geometries
        # TODO: compute OOBB to select the orientation
        visual_geometry = get_mesh_geometry(
            ycb_path, scale=1.0
        )  # TODO: randomly transform
        collision_geometry = get_mesh_geometry(concave_ycb_path, scale=1.0)
        geometry_pose = Pose(point=-mesh.center_mass)
        collision_id = create_collision_shape(
            collision_geometry, pose=geometry_pose, client=client
        )
        visual_id = create_visual_shape(
            visual_geometry, color=color, pose=geometry_pose, client=client
        )
        # collision_id, visual_id = create_shape(geometry, collision=True, color=WHITE)
        body = client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            # basePosition=[0., 0., 0.1]
        )
    elif mesh_type == "default":
        body = create_obj(ycb_path, color=WHITE, client=client, **kwargs)
    elif mesh_type == "hull":
        mesh = read_obj(ycb_path, decompose=False)
        # aabb = aabb_from_points(mesh.vertices)
        # print('Center: {} | Extent: {}'.format(get_aabb_center(aabb), get_aabb_extent(aabb)))
        # draw_aabb(aabb)
        _, _, z = np.min(mesh.vertices, axis=0)
        points = list(mesh.vertices)
        if project_base:  # TODO: base contact points (instead of face) for stability
            points = project_base_points(mesh.vertices, min_z=z, max_z=z + z_threshold)
        hull = mesh_from_points(points)
        print(
            "Name: {} | Mass: {:.3f} | Vertices: {} | Base z: {:.3f} | Hull: {}".format(
                full_name, mass, len(mesh.vertices), z, len(hull.vertices)
            )
        )
        # body = create_faces(hull, mass=mass, color=WHITE, **kwargs)
        body = create_mesh(
            hull, mass=mass, color=WHITE, **kwargs
        )  # TODO: set color to be mean color
        # texture_path = os.path.join(os.path.dirname(ycb_path), 'textured.mtl')
        image_path = os.path.join(os.path.dirname(ycb_path), "texture_map.png")
        texture = p.loadTexture(image_path)
        set_texture(body, texture)
    else:
        raise NotImplementedError(mesh_type)

    set_all_color(body, apply_alpha(color, alpha=1.0), client=client)
    set_grasping_dynamics(body, client=client, **kwargs)
    # dump_body(body)
    # wait_if_gui()
    return Object(
        body, category=name, client=client, **kwargs
    )  # TODO: record the name/color


def create_object(name, mass=STATIC_MASS, color=WHITE, **kwargs):
    if name.endswith("_bowl"):
        path = os.path.join(BOWLS_PATH, "{}.obj".format(name))
        body = create_obj(path, mass=mass, color=color)
    elif name.endswith("_cup"):
        path = os.path.join(CUPS_PATH, "{}.obj".format(name))
        body = create_obj(path, mass=mass, color=color)
    else:
        return create_ycb(name, **kwargs)  # mass=mass, color=color

    set_all_color(body, apply_alpha(color, alpha=1.0))
    set_grasping_dynamics(body, **kwargs)
    return Object(body, category=name, **kwargs)


################################################################################


def set_grasping_dynamics(body, link=BASE_LINK, **kwargs):
    # http://gazebosim.org/tutorials?tut=friction&cat=physics
    # http://gazebosim.org/tutorials?tut=torsional_friction&cat=physics
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf

    # print(get_dynamics_info(body, link))
    # DynamicsInfo(
    #     mass=0.037,
    #     lateral_friction=0.5,
    #     local_inertia_diagonal=(3.629999793842352e-05, 5.8329895304319244e-05, 4.736993218861265e-05),
    #     local_inertial_pos=(0.0, 0.0, 0.0),
    #     local_inertial_orn=(0.0, 0.0, 0.0, 1.0),
    #     restitution=0.0,
    #     rolling_friction=0.0,
    #     spinning_friction=0.0,
    #     contact_damping=-1.0,
    #     contact_stiffness=-1.0
    # )

    # https://github.com/bulletphysics/bullet3/issues/1936
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_data/lego/lego.urdf
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
    set_dynamics(
        body,
        link=link,
        # mass=1.
        lateralFriction=1.0,  # linear (lateral) friction
        spinningFriction=0.1,  # torsional friction around the contact normal
        # rollingFriction=0.01, # torsional friction orthogonal to contact normal
        # restitution=0.01, # restitution: 0 => inelastic collision, 1 => elastic collision
        linearDamping=0.0,  # linear damping of the link
        angularDamping=0.0,  # angular damping of the link
        # localInertiaDiagonal=unit_point(), # diagonal elements of the inertia tensor.
        frictionAnchor=True,  # enable or disable a friction anchor
        contactStiffness=30000.0,  # stiffness of the contact constraints, used together with contactDamping.
        contactDamping=1000.0,  # damping of the contact constraints for this body/link.
        **kwargs
        # Unused fields
        # ccdSweptSphereRadiu=0., # radius of the sphere to perform continuous collision detection.
        # contactProcessingThreshold=0., # contacts with a distance below this threshold will be processed by the constraint solver
        # activationState=0., # When sleeping is enabled, objects that don't move (below a threshold) will be disabled as sleeping
        # jointDamping=0., # Joint damping force = -damping_coefficient * joint_velocity
        # anisotropicFriction=0., # allow scaling of friction in different directions.
        # maxJointVelocity=100., # maximum joint velocity for a given joint, if it is exceeded during constraint solving, it is clamped.
        # collisionMargin=0., # unsupported
        # jointLowerLimit=0., # NOTE that at the moment, the joint limits are not updated in 'getJointInfo'!
        # jointUpperLimit=0., # NOTE that at the moment, the joint limits are not updated in 'getJointInfo'!
        # jointLimitForce=0., # change the maximum force applied to satisfy a joint limit.
    )
    # print(get_dynamics_info(body, link)) # TODO: the default values don't seem to be propagated
    # TODO: set the environment properties


def set_gripper_friction(robot, **kwargs):
    for gripper_group in robot.gripper_groups:
        parent_link = robot.get_group_parent(gripper_group)
        for link in get_link_subtree(robot, parent_link, **kwargs):
            set_grasping_dynamics(robot, link, **kwargs)


################################################################################

BIN = "bin"

SHAPE_INDICES = {
    "tray": [{-1, +1}, {-1, +1}, {-1}],
    BIN: [{-1, +1}, {-1, +1}, {-1}],
    "cubby": [{-1}, {-1, +1}, {-1, +1}],
    "fence": [{-1, +1}, {-1, +1}, {}],
    "x_walls": [[-1, +1], [], [-1]],
    "y_walls": [[], [-1, +1], [-1]],
}

FLOOR_SHAPE = "-z"
CEILING_SHAPE = "+z"


def create_hollow_shapes(indices, width=0.15, length=0.2, height=0.1, thickness=0.01):
    assert len(indices) == 3
    dims = [width, length, height]
    center = [0.0, 0.0, height / 2.0]
    coordinates = string.ascii_lowercase[-len(dims) :]

    # TODO: no way to programmatically set the name of the geoms or links
    # TODO: rigid links version of this
    shapes = []
    for index, signs in enumerate(indices):
        link_dims = np.array(dims)
        link_dims[index] = thickness
        for sign in sorted(signs):
            # name = '{:+d}'.format(sign)
            name = "{}{}".format("-" if sign < 0 else "+", coordinates[index])
            geom = get_box_geometry(*link_dims)
            link_center = np.array(center)
            link_center[index] += sign * (dims[index] - thickness) / 2.0
            pose = Pose(point=link_center)
            shapes.append((name, geom, pose))
    return shapes


def create_hollow(category, color=GREY, *args, **kwargs):
    indices = SHAPE_INDICES[category]
    link = BASE_LINK
    shapes = create_hollow_shapes(indices, *args, **kwargs)
    names, geoms, poses = zip(*shapes)
    shape_indices = list(range(len(geoms)))
    shape_names = {
        name: Shape(link, index) for name, index in zip(names, shape_indices)
    }
    colors = len(shapes) * [color]
    collision_id, visual_id = create_shape_array(geoms, poses, colors)
    body = create_body(collision_id, visual_id, mass=STATIC_MASS)
    return Object(body, category=category, shape_names=shape_names)


def create_tray(*args, **kwargs):
    return create_hollow("tray", *args, **kwargs)


def create_bin(color=GREY, *args, **kwargs):
    return create_hollow(BIN, color=color, *args, **kwargs)


def create_cubby(*args, **kwargs):
    return create_hollow("cubby", *args, **kwargs)


def create_fence(*args, **kwargs):
    return create_hollow("fence", *args, **kwargs)


# TODO: create a grid of regions

################################################################################


def get_grid_cells(rows, columns):
    # TODO: could adjust start row
    return sorted(product(range(0, rows), range(0, columns)))


CUBBY_LINK_TEMPLATE = "[{row},{col}]"
CUBBY_SHAPE_TEMPLATE = CUBBY_LINK_TEMPLATE + "{name}"


def create_cubbies(
    occupied, width=0.15, length=0.2, height=0.1, thickness=0.01, color=GREY
):
    # TODO: use the shelf URDFs from LTAMP
    # TODO: center appropriately
    category = "cubby"
    occupied = sorted(occupied)

    y_step = length - thickness
    z_step = height - thickness

    center_cell = get_aabb_center(aabb_from_points(occupied))
    base_position = Point(y=-y_step * center_cell[1])

    shape_ids = []
    link_poses = []
    link_names = {}
    shape_names = {}
    for link, (row, col) in enumerate(occupied):
        # link_name = 'link{}'.format(link+1)
        link_name = CUBBY_LINK_TEMPLATE.format(row=row, col=col)
        link_names[link_name] = link
        link_poses.append(Pose(Point(y=col * y_step, z=row * z_step)))
        names, geoms, poses = zip(
            *create_hollow_shapes(
                SHAPE_INDICES[category],
                width=width,
                length=length,
                height=height,
                thickness=thickness,
            )
        )
        shape_indices = list(range(len(geoms)))
        shape_names.update(
            {
                CUBBY_SHAPE_TEMPLATE.format(row=row, col=col, name=name): Shape(
                    link, index
                )
                for name, index in zip(names, shape_indices)
            }
        )
        colors = len(geoms) * [color]
        shape_ids.append(
            create_shape_array(geoms, poses, colors)
        )  # limit to number of shapes at once

    collision_ids, visual_ids = zip(*shape_ids)
    link_points = list(map(point_from_pose, link_poses))
    link_quats = list(map(quat_from_pose, link_poses))
    body = p.createMultiBody(
        baseMass=STATIC_MASS,
        baseCollisionShapeIndex=NULL_ID,
        baseVisualShapeIndex=NULL_ID,
        basePosition=base_position,
        baseOrientation=unit_quat(),
        # baseInertialFramePosition=base_position,
        baseInertialFramePosition=unit_point(),
        baseInertialFrameOrientation=unit_quat(),
        linkMasses=len(collision_ids) * [STATIC_MASS],
        linkCollisionShapeIndices=collision_ids,
        linkVisualShapeIndices=visual_ids,
        linkPositions=link_points,
        linkOrientations=link_quats,
        linkInertialFramePositions=len(collision_ids) * [unit_point()],
        linkInertialFrameOrientations=len(collision_ids) * [unit_quat()],
        # linkInertialFramePositions=link_points,
        # linkInertialFramePositions=link_points,
        # linkInertialFrameOrientations=link_quats,
        linkParentIndices=len(collision_ids) * [0],  # BASE_LINK
        linkJointTypes=len(collision_ids) * [p.JOINT_FIXED],
        linkJointAxis=len(collision_ids) * [[0, 0, 0]],
        physicsClientId=CLIENT,
    )
    return Object(
        body, category=category, link_names=link_names, shape_names=shape_names
    )


################################################################################


def create_pillar(width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
    # TODO: use the color when naming
    return Object(
        create_box(w=width, l=length, h=height, color=color, **kwargs),
        category="pillar",
        color=color,
        **kwargs
    )


def create_region(oobb, center, extent, epsilon=1e-3, **kwargs):
    _, upper = oobb.aabb
    z = upper[2] + epsilon / 2.0
    x, y = center
    w, l = extent
    # TODO: could adjust the frame to be consistent with the surface
    pose = multiply(oobb.pose, Pose(Point(x=x, y=y, z=z)))
    body = create_box(w, l, epsilon, mass=STATIC_MASS, **kwargs)
    set_pose(body, pose)
    # TODO: create cylindrical plate
    return Object(body, category="region")


def create_fractional_region(oobb, center_frac, extent_frac, **kwargs):
    # TODO: square version
    lower, upper = oobb.aabb
    # x, y = np.multiply(center_frac, get_aabb_center(oobb.aabb)[:2])
    center = convex_combination(lower[:2], upper[:2], center_frac)
    extent = np.multiply(extent_frac, get_aabb_extent(oobb.aabb)[:2])
    return create_region(oobb, center, extent, **kwargs)


def create_floor_object(color=TAN, **kwargs):
    if color is None:
        body = create_floor(**kwargs)
    else:
        body = create_plane(mass=STATIC_MASS, color=color, **kwargs)
    return Object(body, category="floor", **kwargs)


def create_table_object(color=GREY, **kwargs):
    # body = load_pybullet(TABLE_URDF)
    body = create_table(
        leg_color=color, top_color=color, cylinder=False, mass=STATIC_MASS, **kwargs
    )  # top_color=LIGHT_GREY,
    return Object(body, category="table", **kwargs)


################################################################################


def Pose2D(x=0.0, y=0.0, yaw=0.0):
    return np.array([x, y, yaw])


def place_object(obj, surface, pose2d=Pose2D(), **kwargs):
    surface_oobb = surface.get_shape_oobb()
    z = stable_z_on_aabb(obj, surface_oobb.aabb, **kwargs)  # + Z_EPSILON
    pose = multiply(surface_oobb.pose, pose_from_pose2d(pose2d, z=z))
    set_pose(obj, pose, **kwargs)
    return obj
    # return pose


def place_surface(obj, surface, x=0.0, y=0.35, yaw=0.0, **kwargs):
    # TODO: deprecate
    return place_object(obj, surface, Pose2D(x, y, yaw), **kwargs)


def check_occlude(occluder_obj, occluded_obj, depth_image, seg_image):

    # save_image(os.path.join(TEMP_DIR, 'seg'+str(time.time())+'.png'), seg_image[:, :, 0]) # [0, 255]
    # save_image(os.path.join(TEMP_DIR, 'depth'+str(time.time())+'.png'), depth_image) # [0, 1]

    # Find all of the edge points where there is an object1 pixel next to an object2 pixel
    depths = []
    for i in range(seg_image.shape[0] - 1):
        for j in range(seg_image.shape[1] - 1):
            for edge in [(i + 1, j), (i, j + 1), (i + 1, j + 1)]:
                if (
                    seg_image[i, j, 0] == occluder_obj
                    and seg_image[edge[0], edge[1], 0] == occluded_obj
                ):
                    depths.append(depth_image[edge[0], edge[1]] - depth_image[i, j])

                if (
                    seg_image[i, j, 0] == occluded_obj
                    and seg_image[edge[0], edge[1], 0] == occluder_obj
                ):
                    depths.append(depth_image[i, j] - depth_image[edge[0], edge[1]])

    if len(depths) == 0:
        return False

    # Verify that most of the object1 pixels are smaller distance than all of the object 2 pixels
    return np.mean(np.array(depths)) > 0


def check_fully_occlude(occluded_obj, seg_image):

    # Just need to make sure none of the pixxels in the image are the occluded object
    for i in range(seg_image.shape[0]):
        for j in range(seg_image.shape[1]):
            if seg_image[i, j, 0] == occluded_obj:
                return False
    return True


def get_2d_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def gen_distance_rule(dist=0.1):
    def distance_rule(objs):
        for obj1 in objs:
            for obj2 in objs:
                if (
                    obj1 != obj2
                    and get_2d_dist(get_pose(obj1)[0], get_pose(obj2)[0]) < dist
                ):
                    return False
        return True

    return distance_rule


def no_collision_rule(objs):
    for obj1 in objs:
        for obj2 in objs:
            if obj1 != obj2 and body_collision(obj1, obj2):
                return False
    return True
