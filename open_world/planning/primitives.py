import time
from itertools import chain

import numpy as np
from pybullet_tools.retime import interpolate_path, sample_curve
from pybullet_tools.utils import (
    BASE_LINK,
    INF,
    SEPARATOR,
    Attachment,
    LockRenderer,
    Ray,
    Saver,
    State,
    WorldSaver,
    STATIC_MASS,
    get_mass,
    add_fixed_constraint,
    add_segments,
    adjust_path,
    draw_box_on_image,
    draw_lines_on_image,
    draw_oobb,
    draw_pose,
    elapsed_time,
    empty_sequence,
    flatten,
    get_aabb_edges,
    get_aabb_vertices,
    get_bodies,
    get_closest_points,
    get_fixed_constraints,
    get_joint_positions,
    get_link_pose,
    get_pose,
    get_visible_aabb,
    invert,
    is_fixed_base,
    joint_from_name,
    link_from_name,
    multiply,
    oobb_contains_point,
    pixel_from_ray,
    pose_from_pose2d,
    remove_constraint,
    remove_handles,
    safe_zip,
    set_joint_position,
    set_joint_positions,
    set_pose,
    unit_pose,
    wait_if_gui,
    waypoints_from_path,
    get_joint_names
)

from open_world.estimation.completion import inspect_mesh
from open_world.estimation.observation import (
    draw_labeled_point,
    iterate_point_cloud,
    save_camera_images,
    tform_labeled_points,
)
from open_world.planning.grasping import control_until_contact, get_pregrasp
from open_world.simulation.control import follow_path, stall_for_duration, step_curve
from open_world.simulation.entities import NO_BODY, WORLD_BODY, ParentBody

DRAW_Z = 1e-2
USE_CONSTRAINTS = True
LEAD_CONTROLLER = True


class RelativePose(object):  # TODO: BodyState, RigidAttachment
    # Extends RelPose from SS-Replan
    # Attachment
    def __init__(
        self,
        body,
        parent=None,
        parent_state=None,
        relative_pose=None,
        important=False,
        client=None,
        **kwargs
    ):
        self.body = body
        self.client = client
        # if parent is WORLD_BODY:
        #     parent = ParentBody()
        self.parent = parent
        self.parent_state = parent_state
        if not isinstance(self.body, int):
            self.body = int(str(self.body).split("#")[1])
        if relative_pose is None:
            relative_pose = multiply(
                invert(self.get_parent_pose()), get_pose(self.body, client=self.client)
            )
        self.relative_pose = tuple(relative_pose)
        self.important = important  # TODO: plan harder when true
        # self.initial = False # TODO: initial

    @property
    def value(self):
        return self.relative_pose

    def ancestors(self):
        if self.parent_state is None:
            return [self.body]
        return self.parent_state.ancestors() + [self.body]

    # def ancestors(self):
    #     if self.parent_state is None:
    #         return []
    #     return self.parent_state.ancestors() + [self.parent_state.body]
    def get_parent_pose(self):
        if self.parent is WORLD_BODY:
            return unit_pose()
        if self.parent_state is not None:
            self.parent_state.assign()
        return self.parent.get_pose()

    def get_pose(self):
        return multiply(self.get_parent_pose(), self.relative_pose)

    def assign(self):
        world_pose = self.get_pose()
        set_pose(self.body, world_pose, client=self.client)
        return world_pose

    def draw(self):
        raise NotImplementedError()

    def get_attachment(self):
        assert self.parent is not None
        parent_body, parent_link = self.parent
        # self.assign()
        # return create_attachment(parent_body, parent_link, self.body)
        return Attachment(
            parent_body, parent_link, self.relative_pose, self.body, client=self.client
        )

    def __repr__(self):
        name = "wp" if self.parent is WORLD_BODY else "rp"
        return "{}{}".format(name, id(self) % 1000)


#######################################################


class Grasp(object):  # RelativePose
    def __init__(
        self, body, grasp, pregrasp=None, closed_position=0.0, client=None, **kwargs
    ):
        # TODO: condition on a gripper (or list valid pairs)
        self.body = body
        self.grasp = grasp
        self.client = client
        if pregrasp is None:
            pregrasp = get_pregrasp(grasp)
        self.pregrasp = pregrasp
        self.closed_position = closed_position  # closed_positions

    @property
    def value(self):
        return self.grasp

    @property
    def approach(self):
        return self.pregrasp

    def create_relative_pose(self, robot, link=BASE_LINK):  # create_attachment
        parent = ParentBody(body=robot, link=link, client=self.client)
        return RelativePose(
            self.body, parent=parent, relative_pose=self.grasp, client=self.client
        )

    def create_attachment(self, *args, **kwargs):
        # TODO: create_attachment for a gripper
        relative_pose = self.create_relative_pose(*args, **kwargs)
        return relative_pose.get_attachment()

    def __repr__(self):
        return "g{}".format(id(self) % 1000)


class Conf(object):  # TODO: parent class among Pose, Grasp, and Conf
    # TODO: counter
    def __init__(
        self, body, joints, positions=None, important=False, client=None, **kwargs
    ):
        # TODO: named conf
        self.body = body
        self.joints = joints
        self.client = client
        if positions is None:
            positions = get_joint_positions(self.body, self.joints, client=self.client)
        self.positions = tuple(positions)
        self.important = important
        # TODO: parent state?

    @property
    def robot(self):
        return self.body

    @property
    def values(self):
        return self.positions

    def assign(self):
        set_joint_positions(self.body, self.joints, self.positions, client=self.client)

    def iterate(self):
        yield self

    def __repr__(self):
        return "q{}".format(id(self) % 1000)


class GroupConf(Conf):
    def __init__(self, body, group, *args, **kwargs):
        joints = body.get_group_joints(group)
        super(GroupConf, self).__init__(body, joints, *args, **kwargs)
        self.group = group

    def __repr__(self):
        return "{}q{}".format(self.group[0], id(self) % 1000)


#######################################################


class WorldState(State):
    def __init__(self, savers=[], attachments={}, client=None):
        # a part of the state separate from PyBullet
        # TODO: other fluent things
        super(WorldState, self).__init__(attachments)
        self.world_saver = WorldSaver(client=client)
        self.savers = tuple(savers)
        self.client = client

    def assign(self):
        self.world_saver.restore()
        for saver in self.savers:
            saver.restore()
        self.propagate()

    def copy(self):  # update
        return self.__class__(savers=self.savers, attachments=self.attachments)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, list(self.savers), sorted(self.attachments)
        )


#######################################################


class Command(object):
    # def __init__(self, state=[]):
    #    self.state = tuple(state)

    def switch_client(self):
        raise NotImplementedError

    @property
    def context_bodies(self):
        return set()

    def iterate(self, state, **kwargs):
        raise NotImplementedError()

    def controller(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, controller, *args, **kwargs):
        # raise NotImplementedError()
        return True

    def to_lisdf(self):
        raise NotImplementedError


class BaseSwitch(Command):
    def __init__(self, body, parent=None, client=None, **kwargs):

        self.body = body
        self.parent = parent
        self.client = client

    def iterate(self, state, **kwargs):
        if self.parent is WORLD_BODY and self.body in state.attachments.keys():
            del state.attachments[self.body]
        elif self.parent is not None:
            relative_pose = RelativePose(
                self.body, parent=self.parent, client=self.client
            )
            state.attachments[self.body] = relative_pose

        return empty_sequence()

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class Switch(Command):
    def __init__(self, body, parent=None):
        self.body = body
        self.parent = parent

    def switch_client(self, robot):
        return Switch(self.body, parent=self.parent)

    def iterate(self, state, **kwargs):
        if self.parent is WORLD_BODY and self.body in state.attachments.keys():
            del state.attachments[self.body]
        elif self.parent is not None:
            robot, tool_link = self.parent
            gripper_group = None
            for group, (
                arm_group,
                gripper_group,
                tool_name,
            ) in robot.manipulators.items():
                if link_from_name(robot, tool_name, client=robot.client) == tool_link:
                    break
            else:
                raise RuntimeError(tool_link)
            gripper_joints = robot.get_group_joints(gripper_group)
            finger_links = robot.get_finger_links(gripper_joints)

            movable_bodies = [
                body for body in get_bodies(client=robot.client) if (body != robot)
            ]

            # collision_bodies = [body for body in movable_bodies if any_link_pair_collision(
            #    robot, finger_links, body, max_distance=1e-2)]

            gripper_width = robot.get_gripper_width(gripper_joints)
            max_width = robot.get_max_gripper_width(robot.get_group_joints(gripper_group))

            max_distance = 5e-2
            collision_bodies = [
                body
                for body in movable_bodies
                if (all(
                    get_closest_points(
                        robot,
                        body,
                        link1=link,
                        max_distance=max_distance,
                        client=robot.client,
                    )
                    for link in finger_links
                )  and get_mass(body, client=robot.client) != STATIC_MASS )
            ]

            if len(collision_bodies) > 0:
                relative_pose = RelativePose(
                    collision_bodies[0], parent=self.parent, client=robot.client
                )
                state.attachments[self.body] = relative_pose

        return empty_sequence()

    def controller(self, use_constraints=USE_CONSTRAINTS, **kwargs):
        if not use_constraints:
            return  # empty_sequence()
        if self.parent is WORLD_BODY:
            # TODO: record the robot and tool_link
            for constraint in get_fixed_constraints():
                remove_constraint(constraint)
        else:
            robot, tool_link = self.parent
            gripper_group = None
            for group, (
                arm_group,
                gripper_group,
                tool_name,
            ) in robot.manipulators.items():
                if link_from_name(robot, tool_name) == tool_link:
                    break
            else:
                raise RuntimeError(tool_link)
            gripper_joints = robot.get_group_joints(gripper_group)
            finger_links = robot.get_finger_links(gripper_joints)

            movable_bodies = [
                body
                for body in get_bodies(client=self.robot.client)
                if (body != robot) and not is_fixed_base(body, client=self.robot.client)
            ]
            # collision_bodies = [body for body in movable_bodies if any_link_pair_collision(
            #    robot, finger_links, body, max_distance=1e-2)]

            gripper_width = robot.get_gripper_width(gripper_joints)
            max_distance = gripper_width / 2.0
            collision_bodies = [
                body
                for body in movable_bodies
                if all(
                    get_closest_points(
                        robot, body, link1=link, max_distance=max_distance
                    )
                    for link in finger_links
                )
            ]
            for body in collision_bodies:
                # TODO: improve the PR2's gripper force
                add_fixed_constraint(body, robot, tool_link, max_force=None)
        # TODO: yield for longer
        yield

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)

    def to_lisdf(self):
        return []


class Wait(Command):
    def __init__(self, duration):
        self.duration = duration

    def iterate(self, state, **kwargs):
        return empty_sequence()
        # yield relative_pose

    def controller(self, *args, **kwargs):
        return stall_for_duration(duration=self.duration)
        # return hold_for_duration(robot, duration=self.duration)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.duration)


# class Photograph(Command):
#     def __init__(self, camera, oobb=None):
#         # TODO: make a generic scan the scene command
#         self.camera = camera
#         self.oobb = oobb # TODO: if None, no filtering
#     def iterate(self, state, **kwargs):
#         return empty_sequence()
#     def controller(self, *args, **kwargs):
#         camera_image = self.camera.get_image(segment=False)
#         save_camera_images(camera_image)
#         wait_if_gui()
#         return empty_sequence()
#         #return stall_for_duration(duration=self.duration)
#     def __repr__(self):
#         return '{}({}'.format(self.__class__.__name__, self.camera)


class Inspect(Command):
    def __init__(self, camera, tool_link, inspect_oobbs=[], inspect_trajs=[], known=[]):
        self.camera = camera
        self.tool_link = tool_link
        assert len(inspect_oobbs) == len(inspect_trajs)
        self.inspect_oobbs = tuple(inspect_oobbs)
        self.inspect_trajs = tuple(inspect_trajs)
        # self.known = tuple(sorted_union([NO_BODY, self.robot], known))
        self.known = (NO_BODY, self.robot) + tuple(known)

    @property
    def robot(self):
        return self.camera.robot

    def iterate(self, state, **kwargs):
        return empty_sequence()

    def extract_labeled_points(self, inspect_oobb, draw=False, save=False):
        assert inspect_oobb is not None  # TODO: if None, use the full point cloud
        image_aabb = get_visible_aabb(
            self.camera.camera_matrix, get_aabb_vertices(inspect_oobb.aabb)
        )
        assert image_aabb is not None
        # enable_preview()
        image = self.camera.get_image()

        labeled_points = []
        for labeled_point in iterate_point_cloud(image, aabb=image_aabb, step_size=3):
            body, link = labeled_point.label
            if (body not in self.known) and oobb_contains_point(
                labeled_point.point, inspect_oobb
            ):
                labeled_points.append(labeled_point)
        gripper_pose = get_link_pose(self.robot, self.tool_link)
        labeled_points_gripper = tform_labeled_points(
            invert(gripper_pose), labeled_points
        )

        if draw:
            start_time = time.time()
            handles = draw_oobb(inspect_oobb)
            with LockRenderer():
                for labeled_point in labeled_points_gripper:
                    handles.extend(
                        draw_labeled_point(
                            labeled_point, parent=self.robot, parent_link=self.tool_link
                        )
                    )
                # for labeled_point in labeled_points:
                #    handles.extend(draw_labeled_point(labeled_point))
            print(len(labeled_points_gripper), elapsed_time(start_time))
            wait_if_gui()
            remove_handles(handles)

        if save:
            # TODO: handle case when the drawing is off the image
            for vertices in get_aabb_edges(inspect_oobb.aabb):
                # for vertices in get_wrapped_pairs(support_from_aabb(inspect_oobb.aabb, near=True)):
                pixels = [
                    pixel_from_ray(self.camera.camera_matrix, ray) for ray in vertices
                ]
                image.rgbPixels[...] = draw_lines_on_image(
                    image.rgbPixels, pixels, color="black"
                )
            image.rgbPixels[...] = draw_box_on_image(image.rgbPixels, image_aabb)
            save_camera_images(image)
        return labeled_points_gripper

    def controller(self, *args, **kwargs):
        labeled_points_gripper = []
        for inspect_oobb, inspect_traj in safe_zip(
            self.inspect_oobbs, self.inspect_trajs
        ):
            # for output in Wait(duration=0.25).controller(*args, **kwargs):
            #    yield output
            labeled_points_gripper.extend(self.extract_labeled_points(inspect_oobb))
            for output in inspect_traj.controller(
                *args, **kwargs
            ):  # Wait(duration=0.5)
                yield output
        body = inspect_mesh(labeled_points_gripper, draw=True)
        set_pose(
            body, multiply(get_link_pose(self.robot, self.tool_link), get_pose(body))
        )
        # TODO: create a grasp for the body
        # TODO: incorporate information about the held object
        wait_if_gui()

    def __repr__(self):
        return "{}({}".format(self.__class__.__name__, self.camera)


#######################################################


class Trajectory(Command):
    def __init__(
        self,
        body,
        joints,
        path,
        velocity_scale=1.0,
        contact_links=[],
        time_after_contact=INF,
        contexts=[],
        client=None,
        **kwargs
    ):
        self.body = body
        self.client = client
        self.joints = joints
        self.path = tuple(path)  # waypoints_from_path
        self.velocity_scale = velocity_scale
        self.contact_links = tuple(contact_links)
        self.time_after_contact = time_after_contact
        self.contexts = tuple(contexts)
        # self.kwargs = dict(kwargs) # TODO: doesn't save unpacked values

    # def initialize(self, velocity_scale=1., contact_links=[], time_after_contact=INF, contexts=[]):
    #    pass
    @property
    def robot(self):
        return self.body

    @property
    def context_bodies(self):
        return {self.body} | {
            context.body for context in self.contexts
        }  # TODO: ancestors

    def conf(self, positions):
        return Conf(self.body, self.joints, positions=positions, client=self.client)

    def first(self):
        return self.conf(self.path[0])

    def last(self):
        return self.conf(self.path[-1])

    def reverse(self):
        return self.__class__(
            self.body,
            self.joints,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
        )  # , **self.kwargs)

    def draw(self, only_waypoints=True, **kwargs):
        path = waypoints_from_path(self.path) if only_waypoints else self.path
        handles = []
        if self.group == "base":
            handles.extend(
                draw_pose(pose_from_pose2d(base_conf, z=DRAW_Z), length=5e-2, **kwargs)
                for base_conf in path
            )
        return handles

    def adjust_path(self):
        current_positions = get_joint_positions(
            self.body, self.joints, client=self.client
        )  # Important for adjust_path
        return adjust_path(
            self.body,
            self.joints,
            [current_positions] + list(self.path),
            client=self.client,
        )  # Accounts for the wrap around

    def compute_waypoints(self):
        return waypoints_from_path(
            adjust_path(self.body, self.joints, self.path, client=self.client)
        )

    def compute_curve(self, draw=False, verbose=False, **kwargs):
        path = self.adjust_path()
        # path = self.compute_waypoints()
        # TODO: error when fewer than 2 points
        positions_curve = interpolate_path(
            self.body, self.joints, path, client=self.client
        )
        if verbose:
            print(
                "Following {} {}-DOF waypoints in {:.3f} seconds".format(
                    len(path), len(self.joints), positions_curve.x[-1]
                )
            )
        if not draw:
            return positions_curve
        handles = []
        if self.group == "base":
            # TODO: color by derivative magnitude or theta
            handles.extend(
                add_segments(
                    np.append(q[:2], [DRAW_Z])
                    for t, q in sample_curve(positions_curve, time_step=10.0 / 60)
                )
            )
        wait_if_gui()
        remove_handles(handles)
        return positions_curve

    def traverse(self):
        # TODO: traverse from an initial conf?
        for positions in self.path:
            set_joint_positions(self.body, self.joints, positions)
            yield positions

    def iterate(self, state, teleport=False, **kwargs):
        if(teleport):
            set_joint_positions(self.body, self.joints, self.path[-1], client=self.client)
            return self.path[-1]
        else:
            return step_curve(
                self.body,
                self.joints,
                self.compute_curve(client=self.client, **kwargs),
                client=self.client,
            )

    def controller(self, *args, **kwargs):

        waypoints = self.compute_waypoints()
        if LEAD_CONTROLLER:
            lead_step = 5e-2 * self.velocity_scale
            velocity_scale = None
        else:
            lead_step = None
            velocity_scale = 5e-1  # None | 5e-1
        controller = follow_path(
            self.body,
            self.joints,
            waypoints,
            lead_step=lead_step,
            velocity_scale=velocity_scale,
            max_force=None,
        )  # None | 1e6
        # **self.kwargs)
        # return controller
        return control_until_contact(
            controller, self.body, self.contact_links, self.time_after_contact
        )

    def execute(self, controller, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        return "t{}".format(id(self) % 1000)
        # return '{}x{}'.format(len(self.joints), self.path)


def update_conf(controller, robot, client=None, **kwargs):
    conf = dict(controller.joint_positions)
    for name, position in conf.items():
        joint = joint_from_name(robot, name, client=client)  # TODO: do in batch
        set_joint_position(robot, joint, position, client=client)
    return conf


class GroupTrajectory(Trajectory):
    def __init__(self, body, group, path, *args, **kwargs):
        joints = body.get_group_joints(group)
        super(GroupTrajectory, self).__init__(body, joints, path, *args, **kwargs)
        self.group = group

    def switch_client(self, robot):
        return GroupTrajectory(robot, self.group, self.path, client=robot.client)

    def conf(self, positions):
        return GroupConf(self.body, self.group, positions=positions, client=self.client)

    def execute(self, controller, *args, **kwargs):
        update_conf(controller, self.body)
        velocity_fraction = 0.2
        velocity_fraction *= self.velocity_scale
        positions_curve = self.compute_curve(
            velocity_fraction=velocity_fraction
        )  # TODO: assumes the PyBullet robot is up-to-date
        times, positions = zip(*sample_curve(positions_curve, time_step=1e-1))
        # = np.array(times) / self.velocity_scale
        print(
            "\nGroup: {} | Positions: {} | Duration: {:.3f}\nStart: {}\nEnd: {}".format(
                self.group, len(positions), times[-1], positions[0], positions[-1]
            )
        )
        controller.command_group_trajectory(
            self.group, positions, times, blocking=True, **kwargs
        )
        controller.wait(duration=1.0)
        update_conf(controller, self.body)
        # return True
        if self.group in self.body.gripper_groups:  # Never abort after gripper movement
            return True
        return not controller.any_arm_fully_closed()

    def reverse(self):
        return self.__class__(
            self.body,
            self.group,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
            client=self.client,
        )  # , **self.kwargs)

    def __repr__(self):
        return "{}t{}".format(self.group[0], id(self) % 1000)

    def to_lisdf(self):
        from lisdf.planner_output.command import JointSpacePath, Command, GripperPosition, JointSpacePath, ActuateGripper
        if(self.group in self.robot.gripper_groups):
            closed_conf, open_conf = self.robot.get_group_limits(self.group)
            if( tuple(list(self.path)[-1]) == tuple(closed_conf) ):
                command = ActuateGripper(configurations={"gripper_1": GripperPosition.close}, label=str(self))
            else:
                command = ActuateGripper(configurations={"gripper_1": GripperPosition.open}, label=str(self))
        else:
            joint_names = get_joint_names(self.robot, self.robot.get_group_joints(self.group), client=self.client)
            waypoints = {joint_name: [self.path[i][ji] for i in range(len(self.path))] for ji, joint_name in enumerate(joint_names)}
            command = JointSpacePath(waypoints=waypoints, 
                                     duration=len(waypoints),
                                     label=str(self))

        return [command]
        

#######################################################


class Sequence(Command):  # Commands, CommandSequence
    def __init__(self, commands=[], name=None):
        self.context = None  # TODO: make a State?
        self.commands = tuple(commands)
        self.name = self.__class__.__name__.lower()[:3] if (name is None) else name

    def switch_client(self, robot):
        return Sequence([command.switch_client(robot) for command in self.commands])

    @property
    def context_bodies(self):
        return set(flatten(command.context_bodies for command in self.commands))

    def __len__(self):
        return len(self.commands)

    def iterate(self, *args, **kwargs):
        for command in self.commands:
            print("Executing {} command: {}".format(type(command), str(command)))
            for output in command.iterate(*args, **kwargs):
                yield output

    def controller(self, *args, **kwargs):
        return chain.from_iterable(
            command.controller(*args, **kwargs) for command in self.commands
        )

    def execute(self, *args, return_executed=False, **kwargs):
        executed = []
        for command in self.commands:
            if not command.execute(*args, **kwargs):
                return False, executed if return_executed else False
            executed.append(command)
        return True, executed if return_executed else True

    def reverse(self):
        return Sequence(
            [command.reverse() for command in reversed(self.commands)], name=self.name
        )

    def dump(self):
        print("[{}]".format(" -> ".join(map(repr, self.commands))))

    def __repr__(self):
        return "{}({})".format(self.name, len(self.commands))

    def to_lisdf(self):
        return sum([command.to_lisdf() for command in self.commands], [])

