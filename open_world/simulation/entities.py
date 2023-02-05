from collections import Counter, OrderedDict, namedtuple

import numpy as np
import pybullet as p
from pybullet_tools.pr2_utils import (
    CLEAR_LEFT_ARM,
    COMPACT_LEFT_ARM,
    MAX_KINECT_DISTANCE,
    arm_conf,
    arm_from_side,
    attach_viewcone,
    side_from_arm,
)
from pybullet_tools.utils import (
    BASE_LINK,
    CIRCULAR_LIMITS,
    INF,
    OOBB,
    CameraImage,
    ConfSaver,
    DynamicsInfo,
    JointState,
    Pose,
    get_min_limits,
    get_max_limits,
    get_point,
    set_point,
    get_max_velocities,
    get_unit_vector,
    WorldSaver,
    tform_oobb,
    add_line,
    oobb_from_data,
    TRANSPARENT,
    set_all_color,
    wait_if_gui,
    draw_collision_info,
    RED,
    PoseSaver,
    aabb_from_oobb,
    add_body_name,
    child_link_from_joint,
    clone_body,
    get_aabb,
    get_body_info,
    get_closest_points,
    get_collision_data,
    get_distance,
    get_image_at_pose,
    get_joint_inertial_pose,
    get_joint_info,
    get_joint_parent_frame,
    get_custom_limits,
    get_joints,
    get_link_children,
    get_link_name,
    get_link_pose,
    get_link_state,
    get_link_subtree,
    get_movable_joint_descendants,
    get_movable_joints,
    get_pose,
    invert,
    multiply,
    parent_joint_from_link,
    remove_body,
    remove_handles,
    safe_zip,
    set_joint_positions,
    set_pose,
    unit_pose,
    get_joint_names,
    pixel_from_ray,
    invert,
    multiply,
    dimensions_from_camera_matrix, 
    joint_from_name, 
    set_joint_position
)


# from open_world.simulation.utils import get_color
from open_world.simulation.utils import get_rigid_ancestor

WORLD_BODY = None


class ParentBody(object):  # TODO: inherit from Shape?
    def __init__(
        self, body=WORLD_BODY, link=BASE_LINK, client=None, **kwargs
    ):  # , shape=0):
        self.body = body
        self.client = client
        self.link = link
        # self.shape = shape # shape | index | collision
        # TODO: support surface

    def __iter__(self):
        return iter([self.body, self.link])

    def get_pose(self):
        if self.body is WORLD_BODY:
            return unit_pose()
        return get_link_pose(self.body, self.link, client=self.client)

    # TODO: hash & equals by extending tuple
    def __repr__(self):
        return "Parent({})".format(self.body)


##################################################

# TODO: functions for taking the union of the individual surfaces
defaults = (BASE_LINK, 0)
Shape = namedtuple(
    "Shape", ["link", "index"]
)  # , defaults=defaults) # only supports python3
Shape.__new__.__defaults__ = defaults

DEFAULT_SHAPE = None


class Object(object):
    def __init__(
        self,
        body,
        category=None,
        name=None,
        link_names={},
        shape_names={},
        reference_pose=Pose(),
        color=None,
        properties=[],
        points=[],
        client=None,
        draw=True,
        **kwargs
    ):

        self.client = client
        self.body = body
        self.labeled_points = points
        if category is None:
            category = self.__class__.__name__.lower()
        self.category = category
        if name is None:
            name = "{}#{}".format(self.category, self.body)
        # TODO: could include color & size in name
        self.name = name
        self.link_names = dict(
            link_names
        )  # TODO: only because programmatic creation does not give names
        self.shape_names = dict(shape_names)
        self.reference_pose = tuple(reference_pose)  # TODO: store placement surfaces
        # if color is None:
        #     color = get_color(body)
        self.color = color
        self.properties = list(
            properties
        )  # (Predicate, self, *args) # TODO: could just use self
        self.handles = []
        if draw:
            self.draw()
        # TODO: support faces

    def __int__(self):
        return (
            int(str(self).split("#")[1])
            if not isinstance(self.body, int)
            else self.body
        )

    # def __call__(self, *args, **kwargs):
    #    return self.body
    def __eq__(self, other):
        # TODO: try/except
        return int(self) == int(other)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):  # For heapq on python3
        return int(self) < int(other)

    def __hash__(self):
        return hash(self.body)

    def __repr__(self):
        # return repr(int(self))
        return self.name

    def get_group_parent(self, group):
        # TODO: handle unordered joints
        return self.get_link_parent(self.get_group_joints(group)[0])

    def get_group_subtree(self, group):
        return get_link_subtree(
            self.body, self.get_group_parent(group), client=self.client
        )  # get_link_subtree | get_link_descendants

    def joint_from_name(self, name):
        for joint in get_joints(self.body, client=self.client):
            if self.get_joint_name(joint) == name:
                return joint
        raise ValueError(self.body, name)

    def get_joint_name(self, joint):
        return get_joint_info(self.body, joint, client=self.client).jointName.decode(
            "UTF-8"
        )

    def joints_from_names(self, names):
        return tuple(self.joint_from_name(name) for name in names)

    def get_group_joints(self, group):
        return self.joints_from_names(self.joint_groups[group])

    def get_group_positions(self, group):
        return self.get_joint_positions(self.get_group_joints(group))

    def set_group_positions(self, group, positions):
        set_joint_positions(
            self.body, self.get_group_joints(group), positions, client=self.client
        )

    def get_group_limits(self, group):
        return get_custom_limits(self.body, self.get_group_joints(group), custom_limits=self.custom_limits, client=self.client)

    @property
    def observed_pose(self):
        return get_pose(self.body, client=self.client)

    def is_circular(self, joint):
        joint_info = get_joint_info(self.body, joint, client=self.client)
        if joint_info.jointType == p.JOINT_FIXED:
            return False
        return joint_info.jointUpperLimit < joint_info.jointLowerLimit

    def get_joint_limits(self, joint):
        # TODO: make a version for several joints?
        if self.is_circular(joint):
            # TODO: return UNBOUNDED_LIMITS
            return CIRCULAR_LIMITS
        joint_info = get_joint_info(self.body, joint, client=self.client)
        return joint_info.jointLowerLimit, joint_info.jointUpperLimit


    def get_link_parent(self, link):
        if link == BASE_LINK:
            return None
        return get_joint_info(self.body, link, client=self.client).parentIndex

    def get_local_link_pose(self, joint):
        parent_joint = self.get_link_parent(joint)

        # world_child = get_link_pose(body, joint)
        # world_parent = get_link_pose(body, parent_joint)
        ##return multiply(invert(world_parent), world_child)
        # return multiply(world_child, invert(world_parent))

        # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169
        parent_com = get_joint_parent_frame(self.body, joint, client=self.client)
        tmp_pose = invert(
            multiply(
                get_joint_inertial_pose(self.body, joint, client=self.client),
                parent_com,
            )
        )
        parent_inertia = get_joint_inertial_pose(
            self.body, parent_joint, client=self.client
        )
        # return multiply(parent_inertia, tmp_pose) # TODO: why is this wrong...
        _, orn = multiply(parent_inertia, tmp_pose)
        pos, _ = multiply(parent_inertia, Pose(parent_com[0]))
        return (pos, orn)

    def get_dynamics_info(self, link=BASE_LINK):
        return DynamicsInfo(
            *self.client.getDynamicsInfo(self.body, link)[: len(DynamicsInfo._fields)]
        )

    def get_joint_state(self, joint):
        return JointState(*self.client.getJointState(self.body, joint))

    def get_joint_position(self, joint):
        return self.get_joint_state(joint).jointPosition

    def get_joint_positions(self, joints):  # joints=None):
        return tuple(self.get_joint_position(joint) for joint in joints)

    def get_moving_links(self, joints):
        moving_links = set()
        for joint in joints:
            link = child_link_from_joint(joint)
            if link not in moving_links:
                moving_links.update(
                    get_link_subtree(self.body, link, client=self.client)
                )
        return list(moving_links)

    def get_name(self):
        name = self.get_body_name()
        if name == "":
            name = "body"
        return "{}{}".format(name, int(self.body))

    def get_base_name(self):
        return get_body_info(self.body, client=self.client).base_name.decode(
            encoding="UTF-8"
        )

    def get_body_name(self):
        return get_body_info(self.body, client=self.client).body_name.decode(
            encoding="UTF-8"
        )

    def get_base_name(self):
        return get_body_info(self.body, client=self.client).base_name.decode(
            encoding="UTF-8"
        )

    def link_from_name(self, link_name):
        # return self.link_names.get(link_name, None)
        if link_name in self.link_names:
            return self.link_names[link_name]

        if link_name == self.get_base_name():
            return BASE_LINK
        for link in get_joints(self.body, client=self.client):
            if get_link_name(self.body, link, client=self.client) == link_name:
                return link
        raise ValueError(self.body, link_name)

    def can_collide(self, link=BASE_LINK, **kwargs):
        return len(get_collision_data(self.body, link=link, **kwargs)) != 0

    def get_all_links(self):
        # TODO: deprecate get_links
        return [BASE_LINK] + list(get_joints(self.body))

    @property
    def points(self):
        if isinstance(self.labeled_points[0], tuple):
            return self.labeled_points
        else:
            return [lp.point for lp in self.labeled_points]

    def get_movable_joints(self):
        return self.prune_fixed_joints(get_joints(self.body))

    def get_link_pose(self, link):
        if link == BASE_LINK:
            return get_pose(self.body, client=self.client)
        # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
        link_state = get_link_state(
            self.body, link, client=self.client
        )  # , kinematics=True, velocity=False)
        return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation

    def shape_from_name(self, shape_name):
        return self.shape_names[shape_name]
        # return self.shape_names.get(shape_name, None)

    def get_shape_data(self, link=BASE_LINK, index=0):
        return get_collision_data(self.body, link=link, client=self.client)[index]

    def get_shape_oobb(self, shape_name=DEFAULT_SHAPE):
        # TODO: get_trimesh_oobb
        if shape_name is DEFAULT_SHAPE:
            reference_pose = unit_pose()
            # reference_pose = get_pose(self)
            with PoseSaver(self, client=self.client):
                set_pose(self.body, reference_pose, client=self.client)
                aabb = get_aabb(self.body, client=self.client)
            return OOBB(
                aabb,
                multiply(
                    get_pose(self.body, client=self.client), invert(reference_pose)
                ),
            )  # TODO: double check
            # return OOBB(aabb, Pose())
            # return OOBB(aabb_from_extent_center(get_aabb_extent(aabb)),
            #            Pose(get_aabb_center(aabb)))

        link, index = self.shape_from_name(shape_name)
        surface_data = self.get_shape_data(link, index)
        # pose = get_pose(self)
        pose = get_link_pose(self, link)
        surface_oobb = tform_oobb(pose, oobb_from_data(surface_data))
        # draw_oobb(surface_oobb, color=RED)
        return surface_oobb

    def get_shape_aabb(self, *args, **kwargs):
        return aabb_from_oobb(self.get_shape_oobb(*args, **kwargs))

    @property
    def active(self):
        return self.body is not None

    def erase(self):
        remove_handles(self.handles, client=self.client)

        self.handles = []

    def draw(self):
        self.erase()
        if self.name is not None:
            # TODO: attach to the highest link (for the robot)
            self.handles.append(
                add_body_name(self.body, name=self.name, client=self.client)
            )
        # self.handles.extend(draw_pose(Pose(), parent=self.body, client=self.client))
        return self.handles

    def remove(self):
        self.erase()
        if self.active:
            remove_body(self.body, client=self.client)
            self.body = None


class Table(Object):  # TODO: Region
    def __init__(self, surface, *args, **kwargs):
        self.surface = surface
        super(Table, self).__init__(*args, **kwargs)  # category='table'
        # self.__dict__.update(kwargs)
        # TODO: flag that indicates whether this is a physical region that should avoid collisions

    def draw(self):
        # TODO: other surface related behaviors (e.g. oobb, aabb)
        super(Table, self).draw()
        # self.handles.extend(draw_surface(self.surface))
        return self.handles


##################################################


def simulate_depth(depth_image, min_depth=0.0, max_depth=INF, noise=5e-3):
    if noise > 0:
        depth_image += np.random.normal(scale=noise, size=depth_image.shape)
    depth_image = np.maximum(depth_image, min_depth)
    if max_depth < INF:
        depth_image = np.minimum(depth_image, max_depth)  # TODO: np.nan
    return depth_image


def simulate_noise(camera_image, **kwargs):
    rgb, depth = camera_image[:2]
    # TODO: rgb noise
    depth = simulate_depth(depth, **kwargs)
    return CameraImage(rgb, depth, *camera_image[2:])


class Camera(object):  # TODO: extend Object?
    def __init__(
        self,
        robot,
        link,
        optical_frame,
        camera_matrix,
        max_depth=MAX_KINECT_DISTANCE,
        client=None,
        **kwargs
    ):  # , parent=None): # TODO: remove robot dependency
        self.robot = robot
        self.client = client
        self.link = link  # TODO: no longer need this
        self.optical_frame = optical_frame
        self.camera_matrix = camera_matrix
        self.max_depth = max_depth
        self.kwargs = dict(kwargs)
        # self.parent = parent
        # self.relative_pose = relative_pose
        self.handles = []
        self.draw()

    def get_pose(self):
        return get_link_pose(self.robot, self.optical_frame, client=self.client)

    def get_image(self, segment=True, segment_links=False, **kwargs):
        # TODO: apply maximum depth
        # TODO: noise model
        return get_image_at_pose(
            self.get_pose(),
            self.camera_matrix,
            tiny=False,
            segment=segment,
            segment_links=segment_links,
            client=self.client,
        )  # TODO: OpenCV

    def draw(self, draw_cone=False):
        # draw_link = self.optical_frame # No collision geometery
        # draw_link = self.draw_link

        draw_link = get_rigid_ancestor(
            self.robot, self.optical_frame, client=self.client
        )
        self.robot.get_relative_pose(self.optical_frame, draw_link)
        # draw_pose(multiply(draw_from_optical, Pose()), length=1, parent=self.robot, parent_link=draw_link, client=client)
        if draw_cone:
            attach_viewcone(
                self.robot,
                depth=self.max_depth,
                head_name=get_link_name(self.robot.body, draw_link),
                camera_matrix=self.camera_matrix,
                color=RED,
            )
        # view_cone = get_viewcone(depth=distance, camera_matrix=camera_matrix, color=apply_alpha(RED, alpha=0.1))
        # set_pose(view_cone, camera_pose)
        return self.handles


    def object_visible(self, obj):
        camera_matrix = self.camera_matrix
        obj_pose = get_pose(obj, client=self.client)
        ray = multiply(invert(self.robot.cameras[0].get_pose()), obj_pose)[0]
        image_pixel = pixel_from_ray(camera_matrix, ray)
        width, height = dimensions_from_camera_matrix(self.camera_matrix)
        if(image_pixel[0]<width and image_pixel[0]>=0 and image_pixel[1]<height and image_pixel[1]>=0 and ray[2]>0):
            return True
        return False

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, get_link_name(self.robot.body, self.optical_frame)
        )


##################################################


def invert_dict(d):
    return {v: k for k, v in d.items()}


def map_dict(d, sequence):
    return map(d.get, sequence)


Manipulator = namedtuple(
    "Manipulator", ["arm_group", "gripper_group", "tool_link"]
)  # TODO: gripper_links


class Robot(Object):
    def __init__(
        self,
        body,
        joint_groups={},
        manipulators={},
        cameras=[],
        custom_limits={},
        disabled_collisions={},
        ik_info={},
        joint_weights={},
        joint_resolutions={},
        args=None,
        **kwargs
    ):
        super(Robot, self).__init__(body, **kwargs)

        self.body = body
        self.joint_groups = dict(joint_groups)
        self.manipulators = dict(manipulators)
        self.cameras = tuple(cameras)  # TODO: name the cameras
        self.custom_limits = dict(custom_limits)
        self.disabled_collisions = disabled_collisions
        self.ik_info = dict(ik_info)
        self.joint_weights = dict(joint_weights)
        self.joint_resolutions = dict(joint_resolutions)
        self.components = {}  # grippers


    def update_conf(self, client=None, **kwargs):
        conf = dict(self.controller.joint_positions)
        for name, position in conf.items():
            joint = joint_from_name(self, name, client=client)  # TODO: do in batch
            set_joint_position(self, joint, position, client=client)
        return conf

    def get_relative_pose(self, link1, link2=BASE_LINK):
        world_from_link1 = self.get_link_pose(link1)
        world_from_link2 = self.get_link_pose(link2)
        link2_from_link1 = multiply(invert(world_from_link2), world_from_link1)
        return link2_from_link1
    
    @property
    def default_mobile_base_arm(self):
        return COMPACT_LEFT_ARM

    @property
    def default_fixed_base_arm(self):
        return CLEAR_LEFT_ARM

    @property
    def robot(self):
        return self.body

    @property
    def groups(self):
        return sorted(self.joint_groups)

    @property
    def arm_groups(self):
        return sorted(group for group in self.joint_groups if "arm" in group)

    @property
    def gripper_groups(self):
        return sorted(group for group in self.joint_groups if "gripper" in group)

    @property
    def base_group(self):  # TODO: head group
        group = "base"
        assert group in self.joint_groups
        return group
        
    @property
    def head_group(self):
        return None

    @property
    def base_link(self):
        base_joint = self.get_group_joints(self.base_group)[-1]
        return child_link_from_joint(base_joint)

    def get_gripper_width(self, gripper_joints, draw=False):
        [link1, link2] = self.get_finger_links(gripper_joints)
        [collision_info] = get_closest_points(
            self.body, self.body, link1, link2, max_distance=INF, client=self.client
        )
        point1 = collision_info.positionOnA
        point2 = collision_info.positionOnB
        # distance = collision_info.contactDistance
        if draw:
            draw_collision_info(collision_info)
        max_width = get_distance(point1, point2)
        if draw:
            add_line(point1, point2)
            wait_if_gui()
        return max_width

    def get_max_limit(self, joint):
        return self.get_joint_limits(joint)[1]

    def get_max_limits(self, joints):
        return [self.get_max_limit(joint) for joint in joints]

    def get_max_gripper_width(self, gripper_joints, **kwargs):
        with ConfSaver(self, client=self.client):
            set_joint_positions(
                self.body,
                gripper_joints,
                self.get_max_limits(gripper_joints),
                client=self.client,
            )
            return self.get_gripper_width(gripper_joints, **kwargs)

    def get_finger_links(self, gripper_joints):
        moving_links = self.get_moving_links(gripper_joints)
        shape_links = [
            link
            for link in moving_links
            if get_collision_data(self.body, link, client=self.client)
        ]
        finger_links = [
            link
            for link in shape_links
            if not any(
                get_collision_data(self.body, child, client=self.client)
                for child in get_link_children(self.body, link, client=self.client)
            )
        ]
        # for link in finger_links:
        #     set_color(robot, BLUE, link=link)
        if len(finger_links) != 2:
            raise RuntimeError(finger_links)
        return finger_links

    def get_arbitrary_side(self):
        return sorted(self.manipulators)[0]

    def get_tool_link_pose(self, side):
        arm_group, gripper_group, tool_name = self.manipulators[side]
        tool_link = self.link_from_name(tool_name)
        return self.get_link_pose(tool_link)

    def side_from_arm(self, arm):
        return side_from_arm(arm)

    def arm_from_side(self, side):
        return arm_from_side(side)

    def arm_conf(self, arm, conf):
        return arm_conf(arm, conf)

    def get_component_mapping(self, group):
        # body -> component
        assert group in self.components
        component_joints = get_movable_joints(
            self.components[group], client=self.client, draw=False
        )
        body_joints = get_movable_joint_descendants(
            self.body, self.get_group_parent(group), client=self.client
        )
        return OrderedDict(safe_zip(body_joints, component_joints))

    def get_component_joints(self, group):
        mapping = self.get_component_mapping(group)
        return list(map(mapping.get, self.get_group_joints(group)))

    def get_component_info(self, fn, group):
        return fn(self.body, self.get_group_joints(group))

    def get_component(self, group, visual=True):
        # TODO: ClonedGripper
        # TODO: infinite mass & move away
        # TODO: make a component class
        if group not in self.components:
            component = clone_body(
                self.body,
                links=self.get_group_subtree(group),
                visual=False,
                collision=True,
                client=self.client,
            )
            if not visual:
                set_all_color(component, TRANSPARENT)
            self.components[group] = component
        return self.components[group]

    def remove_components(self):
        for component in self.components.values():
            remove_body(component, client=self.client)
        self.components = {}

    def get_tool_link(self, manipulator):
        _, _, tool_name = self.manipulators[manipulator]
        return self.link_from_name(tool_name)

    def get_parent_from_tool(self, manipulator):
        _, gripper_group, _ = self.manipulators[manipulator]
        tool_link = self.get_tool_link(manipulator)
        parent_link = self.get_group_parent(gripper_group)
        return self.get_relative_pose(tool_link, parent_link)

    def dump(self):
        for group in self.groups:
            joints = self.get_group_joints(group)
            print(get_link_name(self.body, self.get_group_parent(group)))
            print(group, get_joint_names(self.body, joints))


##################################################


class Gripper(object):
    # TODO: update to have the same group structure as Robot
    def __init__(
        self,
        gripper,
        finger_joints=None,
        body=None,
        body_finger_joints=None,
        client=None,
    ):

        if finger_joints is None:
            finger_joints = get_movable_joints(body)
        if body is None:
            body = gripper
        if body_finger_joints is None:
            body_finger_joints = get_movable_joints(body)
        self.gripper = gripper
        self.finger_joints = tuple(finger_joints)
        self.body = body
        self.body_finger_joints = tuple(body_finger_joints)
        # TODO: gripper_from_tool
        # TODO: gripper width

    @property
    def closed_conf(self):
        return get_min_limits(self.body, self.body_finger_joints)

    @property
    def open_conf(self):
        return get_max_limits(self.body, self.body_finger_joints)

    @property
    def max_velocities(self):
        return get_max_velocities(self.body, self.body_finger_joints)

    def get_pose(self):
        return get_pose(self.gripper)

    def set_pose(self, pose):
        set_pose(self.gripper, pose, client=self.client)

    def get_finger_positions(self):
        return self.gripper.get_joint_positions(self.finger_joints)

    def set_finger_positions(self, positions):
        self.gripper.set_joint_positions(self.finger_joints, positions)


class FreeGripper(Gripper):
    def __init__(self, gripper, finger_joints=None):
        super(FreeGripper, self).__init__(
            gripper,
            finger_joints=finger_joints,
            body=gripper,
            body_finger_joints=finger_joints,
        )


class ClonedGripper(Gripper):
    def __init__(self, robot, robot_root_link, robot_finger_joints=None, visual=True):
        self.robot = robot
        self.robot_root_link = robot_root_link

        robot_gripper_links = get_link_subtree(robot, robot_root_link)
        gripper = self.clone_body(
            links=robot_gripper_links, visual=False, collision=True
        )
        if not visual:
            set_all_color(robot, TRANSPARENT)

        robot_gripper_joints = list(map(parent_joint_from_link, robot_gripper_links))
        if robot_finger_joints is None:
            robot_finger_joints = robot_gripper_joints
        robot_finger_indices = [
            idx
            for idx, joint in enumerate(robot_gripper_joints)
            if joint in robot_finger_joints
        ]
        finger_joints = [
            joint
            for joint in get_movable_joints(gripper)
            if joint in robot_finger_indices
        ]
        super(ClonedGripper, self).__init__(
            gripper,
            finger_joints=finger_joints,
            body=robot,
            body_finger_joints=robot_finger_joints,
        )


##################################################


def displace_body(body, vector):
    point = np.array(get_point(body))
    new_point = point + vector
    set_point(body, new_point)
    return new_point


Label = namedtuple("Label", ["category", "instance"])  # TODO: apply to labels

NO_BODY = 256 ** 3 - 1  # TODO: segmented is [16777215.       -2.] when no object is hit
UNKNOWN = "unknown"
TABLE = "table"
BG = "bg"
OTHER = "other"
BOWL = "bowl"  # bowl | 024_bowl
CUP = "cup"  # cup | 025_mug
ENVIRONMENT = [UNKNOWN, TABLE, None]


def get_label_counts(labeled):
    return Counter(tuple(labeled[r, c]) for r, c in np.ndindex(*labeled.shape[:2]))


class RealWorld(object):  # Saver):
    def __init__(
        self,
        robot,
        movable=[],
        fixed=[],
        detectable=[],
        known=[],
        surfaces=[],
        attachable=[],
        materials={},
        displacement=10,
        concave=False,
        client=None,
        room = None,
        **kwargs
    ):
        # TODO: surface class
        self.robot = robot
        self.attachable = attachable
        self.materials = materials
        self.movable = tuple(movable)
        self.fixed = tuple(fixed)  # TODO: infer from is_fixed_base
        self.detectable = tuple(detectable)
        self.concave = concave
        self.known = tuple(known) + (robot,)
        self.surfaces = tuple(surfaces)  # TODO: not used yet
        self.client = client or p
        self.displacement = displacement
        self.saver = None
        self.room = room
        # self.body_savers = [BodySaver(body) for body in get_bodies()]
        # TODO: add/remove new bodies

    @property
    def known_region(self):
        return len(self.surfaces) > 0 or len(self.movable) > 0

    @property
    def objects(self):
        return frozenset(
            self.movable
            + self.fixed
            + self.detectable
            + self.known
            + self.surfaces
            + (self.robot,)
        )

    def disable(self):
        if self.saver is None:
            self.saver = WorldSaver(bodies=self.movable)
            for body in self.movable:
                vector = self.displacement * get_unit_vector([1, 0, 0])
                displace_body(body, vector)
        return self.saver

    def enable(self):
        if self.saver is not None:
            self.saver.restore()
            self.saver = None
        return self.saver

    # def saver(self):
    #    return ClientSaver(self.client)
    # def restore(self):
    #     for body_saver in self.body_savers:
    #         body_saver.restore()
    def label_image(self, camera_image):
        # TODO: make a label class
        obj_from_body = {obj.body: obj for obj in self.objects}
        segmented = camera_image[2].astype(int)
        labeled = np.empty(segmented.shape[:2] + (2,), dtype=object)
        for r in range(segmented.shape[0]):
            for c in range(segmented.shape[1]):
                body, link = segmented[r, c, :]
                if body == NO_BODY:
                    label = Label(None, None)
                elif body in obj_from_body:
                    obj = obj_from_body[body]
                    if obj in self.known:
                        label = Label(obj.category, obj)
                    elif obj in self.detectable:
                        label = Label(obj.category, str(obj.body))
                    else:
                        label = Label(UNKNOWN, UNKNOWN)
                else:
                    label = Label(UNKNOWN, UNKNOWN)
                labeled[r, c] = label
        # print('Ground truth:', get_label_counts(labeled))
        # print('Labels:', get_label_counts(labeled))
        rgb, depth = camera_image[:2]
        return CameraImage(rgb, depth, labeled, *camera_image[3:])
