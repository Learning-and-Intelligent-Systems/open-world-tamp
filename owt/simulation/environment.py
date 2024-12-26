from __future__ import print_function

import math
import os
import string
from itertools import product

import numpy as np
import pybullet as p

import owt.pb_utils as pbu
from owt.estimation.geometry import project_base_points
from owt.simulation.entities import Object, Shape
from owt.simulation.lis import (BOWLS_PATH, CUPS_PATH, YCB_COLORS, YCB_MASSES,
                                get_ycb_obj_path)

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
    concave_ycb_path = get_ycb_obj_path(name, use_concave=use_concave)
    ycb_path = get_ycb_obj_path(name)

    full_name = os.path.basename(os.path.dirname(ycb_path))
    mass = mass_scale * YCB_MASSES[name]

    if (name in YCB_COLORS) and use_concave:
        color = YCB_COLORS[name]
    else:
        color = pbu.WHITE

    print(name, mesh_type, full_name)
    if mesh_type == "drake":
        path = os.path.join(DRAKE_YCB_PATH, "{}.sdf".format(full_name))
        reference_pose = pbu.Pose(euler=pbu.Euler(roll=-np.pi / 2, yaw=-np.pi / 2))
        [body] = pbu.load_pybullet(path)
        pbu.set_pose(body, reference_pose)
    elif mesh_type == "pybullet":
        path = os.path.join(
            PYBULLET_YCB_DIR,
            "Ycb{}/model.urdf".format(
                "".join(word.capitalize() for word in name.split("_"))
            ),
        )
        body = pbu.load_pybullet(path)
        reference_pose = pbu.Pose(euler=pbu.Euler(yaw=0))
        pbu.set_pose(body, reference_pose)
    elif mesh_type == "centered":
        import trimesh

        mesh = trimesh.load(ycb_path)
        visual_geometry = pbu.get_mesh_geometry(
            ycb_path, scale=1.0
        )  # TODO: randomly transform
        collision_geometry = pbu.get_mesh_geometry(concave_ycb_path, scale=1.0)
        geometry_pose = pbu.Pose(point=-mesh.center_mass)
        collision_id = pbu.create_collision_shape(
            collision_geometry, pose=geometry_pose, client=client
        )
        visual_id = pbu.create_visual_shape(
            visual_geometry, color=color, pose=geometry_pose, client=client
        )
        body = client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
        )
    elif mesh_type == "default":
        body = pbu.create_obj(ycb_path, color=pbu.WHITE, client=client, **kwargs)
    elif mesh_type == "hull":
        mesh = pbu.read_obj(ycb_path, decompose=False)
        _, _, z = np.min(mesh.vertices, axis=0)
        points = list(mesh.vertices)
        if project_base:  # TODO: base contact points (instead of face) for stability
            points = project_base_points(mesh.vertices, min_z=z, max_z=z + z_threshold)
        hull = pbu.mesh_from_points(points)
        print(
            "Name: {} | Mass: {:.3f} | Vertices: {} | Base z: {:.3f} | Hull: {}".format(
                full_name, mass, len(mesh.vertices), z, len(hull.vertices)
            )
        )
        body = pbu.create_mesh(hull, mass=mass, color=pbu.WHITE, **kwargs)
        image_path = os.path.join(os.path.dirname(ycb_path), "texture_map.png")
        texture = p.loadTexture(image_path)
        pbu.set_texture(body, texture)
    else:
        raise NotImplementedError(mesh_type)

    pbu.set_all_color(body, pbu.apply_alpha(color, alpha=1.0), client=client)
    set_grasping_dynamics(body, client=client, **kwargs)
    # dump_body(body)
    # wait_if_gui()
    return Object(
        body, category=name, client=client, **kwargs
    )  # TODO: record the name/color


def create_object(name, mass=pbu.STATIC_MASS, color=pbu.WHITE, **kwargs):
    if name.endswith("_bowl"):
        path = os.path.join(BOWLS_PATH, "{}.obj".format(name))
        body = pbu.create_obj(path, mass=mass, color=color)
    elif name.endswith("_cup"):
        path = os.path.join(CUPS_PATH, "{}.obj".format(name))
        body = pbu.create_obj(path, mass=mass, color=color)
    else:
        return create_ycb(name, **kwargs)  # mass=mass, color=color

    pbu.set_all_color(body, pbu.apply_alpha(color, alpha=1.0))
    set_grasping_dynamics(body, **kwargs)
    return Object(body, category=name, **kwargs)


################################################################################


def set_grasping_dynamics(body, link=pbu.BASE_LINK, **kwargs):
    pbu.set_dynamics(
        body,
        link=link,
        lateralFriction=1.0,  # linear (lateral) friction
        spinningFriction=0.1,  # torsional friction around the contact normal
        linearDamping=0.0,  # linear damping of the link
        angularDamping=0.0,  # angular damping of the link
        frictionAnchor=True,  # enable or disable a friction anchor
        contactStiffness=30000.0,  # stiffness of the contact constraints, used together with contactDamping.
        contactDamping=1000.0,  # damping of the contact constraints for this body/link.
        **kwargs
    )


def set_gripper_friction(robot, **kwargs):
    for gripper_group in robot.gripper_groups:
        parent_link = robot.get_group_parent(gripper_group)
        for link in pbu.get_link_subtree(robot, parent_link, **kwargs):
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
            geom = pbu.get_box_geometry(*link_dims)
            link_center = np.array(center)
            link_center[index] += sign * (dims[index] - thickness) / 2.0
            pose = pbu.Pose(point=link_center)
            shapes.append((name, geom, pose))
    return shapes


def create_hollow(category, color=pbu.GREY, *args, **kwargs):
    indices = SHAPE_INDICES[category]
    link = pbu.BASE_LINK
    shapes = create_hollow_shapes(indices, *args, **kwargs)
    names, geoms, poses = zip(*shapes)
    shape_indices = list(range(len(geoms)))
    shape_names = {
        name: Shape(link, index) for name, index in zip(names, shape_indices)
    }
    colors = len(shapes) * [color]
    collision_id, visual_id = pbu.create_shape_array(geoms, poses, colors)
    body = pbu.create_body(collision_id, visual_id, mass=pbu.STATIC_MASS)
    return Object(body, category=category, shape_names=shape_names)


def create_tray(*args, **kwargs):
    return create_hollow("tray", *args, **kwargs)


def create_bin(color=pbu.GREY, *args, **kwargs):
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
    occupied, width=0.15, length=0.2, height=0.1, thickness=0.01, color=pbu.GREY
):
    category = "cubby"
    occupied = sorted(occupied)

    y_step = length - thickness
    z_step = height - thickness

    center_cell = pbu.get_aabb_center(pbu.aabb_from_points(occupied))
    base_position = pbu.Point(y=-y_step * center_cell[1])

    shape_ids = []
    link_poses = []
    link_names = {}
    shape_names = {}
    for link, (row, col) in enumerate(occupied):
        link_name = CUBBY_LINK_TEMPLATE.format(row=row, col=col)
        link_names[link_name] = link
        link_poses.append(pbu.Pose(pbu.Point(y=col * y_step, z=row * z_step)))
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
        shape_ids.append(pbu.create_shape_array(geoms, poses, colors))

    collision_ids, visual_ids = zip(*shape_ids)
    link_points = list(map(pbu.point_from_pose, link_poses))
    link_quats = list(map(pbu.quat_from_pose, link_poses))
    body = p.createMultiBody(
        baseMass=pbu.STATIC_MASS,
        baseCollisionShapeIndex=pbu.NULL_ID,
        baseVisualShapeIndex=pbu.NULL_ID,
        basePosition=base_position,
        baseOrientation=pbu.unit_quat(),
        baseInertialFramePosition=pbu.unit_point(),
        baseInertialFrameOrientation=pbu.unit_quat(),
        linkMasses=len(collision_ids) * [pbu.STATIC_MASS],
        linkCollisionShapeIndices=collision_ids,
        linkVisualShapeIndices=visual_ids,
        linkPositions=link_points,
        linkOrientations=link_quats,
        linkInertialFramePositions=len(collision_ids) * [pbu.unit_point()],
        linkInertialFrameOrientations=len(collision_ids) * [pbu.unit_quat()],
        linkParentIndices=len(collision_ids) * [0],
        linkJointTypes=len(collision_ids) * [p.JOINT_FIXED],
        linkJointAxis=len(collision_ids) * [[0, 0, 0]],
    )
    return Object(
        body, category=category, link_names=link_names, shape_names=shape_names
    )


################################################################################


def create_pillar(width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
    return Object(
        pbu.create_box(w=width, l=length, h=height, color=color, **kwargs),
        category="pillar",
        color=color,
        **kwargs
    )


def create_region(oobb, center, extent, epsilon=1e-3, **kwargs):
    _, upper = oobb.aabb
    z = upper[2] + epsilon / 2.0
    x, y = center
    w, l = extent
    pose = pbu.multiply(oobb.pose, pbu.Pose(pbu.Point(x=x, y=y, z=z)))
    body = pbu.create_box(w, l, epsilon, mass=pbu.STATIC_MASS, **kwargs)
    pbu.set_pose(body, pose)
    return Object(body, category="region")


def create_fractional_region(oobb, center_frac, extent_frac, **kwargs):
    lower, upper = oobb.aabb
    center = pbu.convex_combination(lower[:2], upper[:2], center_frac)
    extent = np.multiply(extent_frac, pbu.get_aabb_extent(oobb.aabb)[:2])
    return create_region(oobb, center, extent, **kwargs)


def create_floor(**kwargs):
    FLOOR_URDF = "plane.urdf"
    pbu.add_data_path()
    return pbu.load_pybullet(FLOOR_URDF, **kwargs)


def create_plane(normal=[0, 0, 1], mass=pbu.STATIC_MASS, color=pbu.BLACK, **kwargs):
    # color seems to be ignored in favor of a texture
    collision_id, visual_id = pbu.create_shape(
        pbu.get_plane_geometry(normal), color=color, **kwargs
    )
    body = pbu.create_body(collision_id, visual_id, mass=mass, **kwargs)
    pbu.set_texture(body, texture=None, **kwargs)  # otherwise 'plane.urdf'
    pbu.set_color(body, color=color, **kwargs)  # must perform after set_texture
    return body


def create_floor_object(color=pbu.TAN, **kwargs):
    if color is None:
        body = create_floor(**kwargs)
    else:
        body = create_plane(mass=pbu.STATIC_MASS, color=color, **kwargs)
    return Object(body, category="floor", **kwargs)


def create_table(
    width=0.6,
    length=1.2,
    height=0.73,
    thickness=0.03,
    radius=0.015,
    top_color=pbu.LIGHT_GREY,
    leg_color=pbu.TAN,
    cylinder=True,
    **kwargs
):
    surface = pbu.get_box_geometry(width, length, thickness)
    surface_pose = pbu.Pose(pbu.Point(z=height - thickness / 2.0))

    leg_height = height - thickness
    if cylinder:
        leg_geometry = pbu.get_cylinder_geometry(radius, leg_height)
    else:
        leg_geometry = pbu.get_box_geometry(
            width=2 * radius, length=2 * radius, height=leg_height
        )
    legs = [leg_geometry for _ in range(4)]
    leg_center = np.array([width, length]) / 2.0 - radius * np.ones(2)
    leg_xys = [
        np.multiply(leg_center, np.array(signs))
        for signs in product([-1, +1], repeat=len(leg_center))
    ]
    leg_poses = [pbu.Pose(point=[x, y, leg_height / 2.0]) for x, y in leg_xys]

    geoms = [surface] + legs
    poses = [surface_pose] + leg_poses
    colors = [top_color] + len(legs) * [leg_color]

    collision_id, visual_id = pbu.create_shape_array(geoms, poses, colors, **kwargs)
    body = pbu.create_body(collision_id, visual_id, **kwargs)

    return body


def create_table_object(color=pbu.GREY, **kwargs):
    body = create_table(
        leg_color=color, top_color=color, cylinder=False, mass=pbu.STATIC_MASS, **kwargs
    )
    return Object(body, category="table", **kwargs)


################################################################################


def Pose2D(x=0.0, y=0.0, yaw=0.0):
    return np.array([x, y, yaw])


def place_object(obj, surface, pose2d=Pose2D(), **kwargs):
    surface_oobb = surface.get_shape_oobb()
    z = pbu.stable_z_on_aabb(obj, surface_oobb.aabb, **kwargs)  # + Z_EPSILON
    pose = pbu.multiply(surface_oobb.pose, pbu.pose_from_pose2d(pose2d, z=z))
    pbu.set_pose(obj, pose, **kwargs)
    return obj


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
                    and get_2d_dist(pbu.get_pose(obj1)[0], pbu.get_pose(obj2)[0]) < dist
                ):
                    return False
        return True

    return distance_rule


def no_collision_rule(objs):
    for obj1 in objs:
        for obj2 in objs:
            if obj1 != obj2 and pbu.body_collision(obj1, obj2):
                return False
    return True
