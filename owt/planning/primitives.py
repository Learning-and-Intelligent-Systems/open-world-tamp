import itertools

import numpy as np

import owt.pb_utils as pbu
from owt.planning.grasping import control_until_contact, get_pregrasp
from owt.simulation.control import follow_path, stall_for_duration, step_curve
from owt.simulation.entities import WORLD_BODY, ParentBody

DRAW_Z = 1e-2
USE_CONSTRAINTS = True
LEAD_CONTROLLER = True


class RelativePose(object):
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
        self.parent = parent
        self.parent_state = parent_state
        if not isinstance(self.body, int):
            self.body = int(str(self.body).split("#")[1])
        if relative_pose is None:
            relative_pose = pbu.multiply(
                pbu.invert(self.get_parent_pose()),
                pbu.get_pose(self.body, client=self.client),
            )
        self.relative_pose = tuple(relative_pose)
        self.important = important  # TODO: plan harder when true

    @property
    def value(self):
        return self.relative_pose

    def ancestors(self):
        if self.parent_state is None:
            return [self.body]
        return self.parent_state.ancestors() + [self.body]

    def get_parent_pose(self):
        if self.parent is WORLD_BODY:
            return pbu.unit_pose()
        if self.parent_state is not None:
            self.parent_state.assign()
        return self.parent.get_pose()

    def get_pose(self):
        return pbu.multiply(self.get_parent_pose(), self.relative_pose)

    def assign(self):
        world_pose = self.get_pose()
        pbu.set_pose(self.body, world_pose, client=self.client)
        return world_pose

    def draw(self):
        raise NotImplementedError()

    def get_attachment(self):
        assert self.parent is not None
        parent_body, parent_link = self.parent
        return pbu.Attachment(
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

    def create_relative_pose(self, robot, link=pbu.BASE_LINK):  # create_attachment
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
            positions = pbu.get_joint_positions(
                self.body, self.joints, client=self.client
            )
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
        pbu.set_joint_positions(
            self.body, self.joints, self.positions, client=self.client
        )

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


class WorldState(pbu.State):
    def __init__(self, savers=[], attachments={}, client=None):
        # a part of the state separate from PyBullet
        # TODO: other fluent things
        super(WorldState, self).__init__(attachments)
        self.world_saver = pbu.WorldSaver(client=client)
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


class Command:
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

        return pbu.empty_sequence()

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
            for _, (_, gripper_group, tool_name) in robot.manipulators.items():
                if (
                    pbu.link_from_name(robot, tool_name, client=robot.client)
                    == tool_link
                ):
                    break
            else:
                raise RuntimeError(tool_link)
            gripper_joints = robot.get_group_joints(gripper_group)
            finger_links = robot.get_finger_links(gripper_joints)

            movable_bodies = [
                body for body in pbu.get_bodies(client=robot.client) if (body != robot)
            ]

            max_distance = 5e-2
            collision_bodies = [
                body
                for body in movable_bodies
                if (
                    all(
                        pbu.get_closest_points(
                            robot,
                            body,
                            link1=link,
                            max_distance=max_distance,
                            client=robot.client,
                        )
                        for link in finger_links
                    )
                    and pbu.get_mass(body, client=robot.client) != pbu.STATIC_MASS
                )
            ]

            if len(collision_bodies) > 0:
                relative_pose = RelativePose(
                    collision_bodies[0], parent=self.parent, client=robot.client
                )
                state.attachments[self.body] = relative_pose

        return pbu.empty_sequence()

    def controller(self, use_constraints=USE_CONSTRAINTS, **kwargs):
        if not use_constraints:
            return
        if self.parent is WORLD_BODY:
            for constraint in pbu.get_fixed_constraints():
                pbu.remove_constraint(constraint)
        else:
            robot, tool_link = self.parent
            gripper_group = None
            for group, (
                arm_group,
                gripper_group,
                tool_name,
            ) in robot.manipulators.items():
                if pbu.link_from_name(robot, tool_name) == tool_link:
                    break
            else:
                raise RuntimeError(tool_link)
            gripper_joints = robot.get_group_joints(gripper_group)
            finger_links = robot.get_finger_links(gripper_joints)

            movable_bodies = [
                body
                for body in pbu.get_bodies(client=self.robot.client)
                if (body != robot)
                and not pbu.is_fixed_base(body, client=self.robot.client)
            ]
            gripper_width = robot.get_gripper_width(gripper_joints)
            max_distance = gripper_width / 2.0
            collision_bodies = [
                body
                for body in movable_bodies
                if all(
                    pbu.get_closest_points(
                        robot, body, link1=link, max_distance=max_distance
                    )
                    for link in finger_links
                )
            ]
            for body in collision_bodies:
                pbu.add_fixed_constraint(body, robot, tool_link, max_force=None)
        yield

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class Wait(Command):
    def __init__(self, duration):
        self.duration = duration

    def iterate(self, state, **kwargs):
        return pbu.empty_sequence()

    def controller(self, *args, **kwargs):
        return stall_for_duration(duration=self.duration)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.duration)


class Trajectory(Command):
    def __init__(
        self,
        robot,
        joints,
        path,
        velocity_scale=1.0,
        contact_links=[],
        time_after_contact=np.inf,
        contexts=[],
        client=None,
        **kwargs
    ):
        self.robot = robot
        self.client = client
        self.joints = joints
        self.path = tuple(path)  # waypoints_from_path
        self.velocity_scale = velocity_scale
        self.contact_links = tuple(contact_links)
        self.time_after_contact = time_after_contact
        self.contexts = tuple(contexts)

    @property
    def context_bodies(self):
        return {self.robot} | {
            context.body for context in self.contexts if hasattr(context, "body")
        }

    def conf(self, positions):
        return Conf(self.robot, self.joints, positions=positions, client=self.client)

    def first(self):
        return self.conf(self.path[0])

    def last(self):
        return self.conf(self.path[-1])

    def reverse(self):
        return self.__class__(
            self.robot,
            self.joints,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
        )  # , **self.kwargs)

    def draw(self, only_waypoints=True, **kwargs):
        path = pbu.waypoints_from_path(self.path) if only_waypoints else self.path
        handles = []
        if self.group == "base":
            handles.extend(
                pbu.draw_pose(
                    pbu.pose_from_pose2d(base_conf, z=DRAW_Z), length=5e-2, **kwargs
                )
                for base_conf in path
            )
        return handles

    def adjust_path(self):
        current_positions = pbu.get_joint_positions(
            self.robot, self.joints, client=self.client
        )  # Important for adjust_path
        return pbu.adjust_path(
            self.robot,
            self.joints,
            [current_positions] + list(self.path),
            client=self.client,
        )  # Accounts for the wrap around

    def compute_waypoints(self):
        return pbu.waypoints_from_path(
            pbu.adjust_path(self.robot, self.joints, self.path, client=self.client)
        )

    def compute_curve(self, draw=False, verbose=False, **kwargs):
        path = self.adjust_path()
        positions_curve = pbu.interpolate_path(
            self.robot, self.joints, path, client=self.client
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
            handles.extend(
                pbu.add_segments(
                    np.append(q[:2], [DRAW_Z])
                    for t, q in pbu.sample_curve(positions_curve, time_step=10.0 / 60)
                )
            )
        pbu.wait_if_gui()
        pbu.remove_handles(handles)
        return positions_curve

    def traverse(self):
        for positions in self.path:
            pbu.set_joint_positions(self.robot, self.joints, positions)
            yield positions

    def iterate(self, state, teleport=False, **kwargs):
        if teleport:
            pbu.set_joint_positions(
                self.robot, self.joints, self.path[-1], client=self.client
            )
            return self.path[-1]
        else:
            return step_curve(
                self.robot,
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
            self.robot,
            self.joints,
            waypoints,
            lead_step=lead_step,
            velocity_scale=velocity_scale,
            max_force=None,
        )  # None | 1e6
        return control_until_contact(
            controller, self.robot, self.contact_links, self.time_after_contact
        )

    def execute(self, controller, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        return "t{}".format(id(self) % 1000)


class GroupTrajectory(Trajectory):
    def __init__(self, robot, group, path, *args, **kwargs):
        joints = robot.get_group_joints(group)
        super(GroupTrajectory, self).__init__(robot, joints, path, *args, **kwargs)
        self.group = group

    def switch_client(self, robot):
        return GroupTrajectory(robot, self.group, self.path, client=robot.client)

    def conf(self, positions):
        return GroupConf(
            self.robot, self.group, positions=positions, client=self.client
        )

    def execute(self, controller, *args, **kwargs):
        self.robot.update_conf()
        velocity_fraction = 0.2
        velocity_fraction *= self.velocity_scale
        positions_curve = self.compute_curve(
            velocity_fraction=velocity_fraction
        )  # TODO: assumes the PyBullet robot is up-to-date
        times, positions = zip(*pbu.sample_curve(positions_curve, time_step=1e-1))
        print(
            "\nGroup: {} | Positions: {} | Duration: {:.3f}\nStart: {}\nEnd: {}".format(
                self.group, len(positions), times[-1], positions[0], positions[-1]
            )
        )
        controller.command_group_trajectory(
            self.group, positions, times, blocking=True, **kwargs
        )
        controller.wait(duration=1.0)
        self.robot.update_conf()
        if (
            self.group in self.robot.gripper_groups
        ):  # Never abort after gripper movement
            return True
        return not controller.any_arm_fully_closed()

    def reverse(self):
        return self.__class__(
            self.robot,
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
        return set(
            itertools.chain(*[command.context_bodies for command in self.commands])
        )

    def __len__(self):
        return len(self.commands)

    def iterate(self, *args, **kwargs):
        for command in self.commands:
            print("Executing {} command: {}".format(type(command), str(command)))
            for output in command.iterate(*args, **kwargs):
                yield output

    def controller(self, *args, **kwargs):
        return itertools.chain.from_iterable(
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
