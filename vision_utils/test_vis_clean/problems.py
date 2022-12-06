from __future__ import print_function

import numpy as np
from examples.discrete_belief.dist import DeltaDist, MixtureDD, MixtureDist, UniformDist
from examples.pybullet.utils.pybullet_tools.pr2_primitives import State
from examples.pybullet.utils.pybullet_tools.pr2_problems import (
    create_kitchen,
    create_pr2,
)
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    REST_LEFT_ARM,
    arm_conf,
    close_arm,
    create_gripper,
    get_carry_conf,
    get_other_arm,
    open_arm,
    set_arm_conf,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    HideOutput,
    get_bodies,
    get_name,
    is_center_stable,
)

USE_DRAKE_PR2 = True
OTHER = "other"
LOCALIZED_PROB = 0.99

""" ========================  modified/added classes ======================== """


class aPose(object):
    def __init__(self, body, value=None, support=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.init = init
        self.label = {}
        self.label_final = None
        self.uncertainty = 0

    def assign(self):
        set_pose(self.body, self.value)

    def iterate(self):
        yield self

    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)

    def __repr__(self):
        return "p{}".format(id(self) % 1000)


class Voxelgrid(object):
    def __init__(self, resolution, lower, upper):
        self.occupied = set()  # unobserved
        self.sampled = {}
        self.box = {}
        self.lower = lower
        self.upper = upper
        self.resolution = resolution
        self.max = self.max_voxel()
        self.vis = False  # visualize the boxes or not
        self.poses = {}
        self.voxels = []
        for i in range(int(self.max[0])):
            for j in range(int(self.max[1])):
                for k in range(int(self.max[2])):
                    self.voxels.append([i, j, k])
                    self.set_occupied((i, j, k), vis=self.vis)
                    self.poses[(i, j, k)] = aPose(
                        body=None,
                        value=(self.center_from_voxel((i, j, k)), (0, 0, 0, 1)),
                    )

    def copy(self):
        new_grid = Voxelgrid(self.resolution, self.lower, self.upper)
        new_grid.occupied = self.occupied.copy()
        # new_grid.box = self.box.deepcopy()
        return new_grid

    def fully_observed(self):
        # max_v = self.max_voxel()
        return len(self.occupied) == 0  # (max_v[0]*max_v[1]*max_v[2])

    def point_in_space(self, point, threshold=(0.2, 0.2, 0.2)):
        return all_between(
            np.subtract(self.lower, threshold), point, np.add(self.upper, threshold)
        )

    def voxel_from_point(self, point):
        """
        index of voxel for the target point
        """
        if not self.point_in_space(point):
            return -1
        return np.floor(np.subtract(point, self.lower) / self.resolution).astype(np.int)

    def voxels_from_aabb(self, aabb):
        lower_voxel, upper_voxel = map(self.voxel_from_point, aabb)
        voxels = map(
            tuple,
            product(*[range(l, u + 1) for l, u in safe_zip(lower_voxel, upper_voxel)]),
        )
        return list(voxels)

    def lower_from_voxel(self, voxel):
        return np.multiply(voxel, self.resolution) + self.lower

    def center_from_voxel(self, voxel):
        return np.multiply(voxel, self.resolution) + self.resolution / 2.0 + self.lower

    def upper_from_voxel(self, voxel):
        return np.multiply(voxel, self.resolution) + self.resolution + self.lower

    def aabb_from_voxel(self, voxel):
        return AABB(self.lower_from_voxel(voxel), self.upper_from_voxel(voxel))

    def max_voxel(self):
        return np.ceil(
            np.subtract(self.upper, self.lower) / self.resolution
        )  # self.voxel_from_point(self.upper) - (1,1,1)

    def hide_vblock(self):
        for voxel in self.occupied:
            # if voxel in self.box.keys():
            set_color(self.box[voxel], np.zeros(4))

    def show_vblock(self):
        for voxel in self.occupied:
            # if voxel in self.box.keys():
            set_color(self.box[voxel], (1, 0, 0, 0.3))

    def is_occupied(self, voxel):
        return voxel in self.occupied

    def set_occupied(self, voxel, vis=False):
        # if voxel not in self.box.keys():
        #     self.box[voxel] = self.create_box(self.center_from_voxel(voxel),(0,0,0,0))
        if not self.is_occupied(voxel):
            self.occupied.add(voxel)
            if vis:
                set_color(self.box[voxel], (1, 0, 0, 0.3))

    def set_occupieds(self, voxels, vis=False):
        for voxel in voxels:
            self.set_occupied(voxel, vis=vis)

    def set_free(self, voxel, vis=False):
        if self.is_occupied(voxel):
            self.occupied.remove(voxel)
            if vis:
                # if voxel in self.box.keys():
                set_color(self.box[voxel], np.zeros(4))

    def set_frees(self, voxels, vis=False):
        for voxel in voxels:
            self.set_free(voxel, vis=vis)

    # def create_virtual_box(self, pose=unit_pose(), color=np.zeros(4)):
    #     # color = (1,0,1,0.3)
    #     box = create_virtual_box(self.resolution, self.resolution, self.resolution, color=color)
    #     set_point(box, pose)
    #     return box

    # def create_box(self, pose=unit_pose(), color=np.zeros(4)):
    #     # color = (1,0,1,0.3)
    #     box = create_box(self.resolution, self.resolution, self.resolution, color=color)
    #     set_pose(box, pose)
    #     return box

    def draw_voxel(self):  # deprecated?
        with LockRenderer():
            handles = []
            for voxel in sorted(self.occupied):
                handles.extend(
                    draw_aabb(self.aabb_from_voxel(voxel), color=(0, 0.5, 0))
                )
            return handles

    def get_coarse_affected(self, body):
        aabb = get_aabb(body)
        affected = self.voxels_from_aabb(aabb)
        max_voxel = np.asarray(self.max_voxel())
        effect_affected = [
            voxel
            for voxel in affected
            if all(np.asarray(voxel) >= np.zeros(3))
            and all(np.asarray(voxel) < max_voxel)
        ]
        return effect_affected

    def check_vis(
        self,
        camera_pos,
        eye_coord_cloud,
        segment,
        view_matrix_real,
        proj_matrix_real,
        target_point,
        threshold=0.0,
        entity_list=[],
    ):
        """
        params:
            camera_pose: eye position
            target_point: target position
        return:

        """
        block_list = {}
        can_clear = []
        # invalid_list = [-1,0,1] # TODO adhoc. invalid objects in maskrcnn return value [x deprecated get_image.segment, use maskrcnn instead of pybullet segmenter now]
        # object_list = {e.category:e for e in entity_list if e.obj_id not in
        #         invalid_list}
        # # print(entity_list,object_list)
        object_list = {e.category + 2: e for e in entity_list}
        base_pose = camera_pos
        diff = np.asarray(list(target_point)) - np.asarray(list(base_pose))
        diff_len = (diff ** 2).sum() ** 0.5

        for voxel in self.occupied:
            center_loc = np.asarray(self.center_from_voxel(voxel))
            im_loc = proj_matrix_real.dot(
                view_matrix_real.dot(np.concatenate((center_loc, np.ones(1))))
            )
            im_x_n = im_loc[0] / im_loc[3]
            im_y_n = im_loc[1] / im_loc[3]
            if im_x_n >= -1 and im_x_n <= 1 and im_y_n >= -1 and im_y_n <= 1:
                im_x = int((im_x_n + 1.0) / 2.0 * 640)
                im_y = int((-im_y_n + 1.0) / 2.0 * 480)
                eye_depth = (eye_coord_cloud[im_y, im_x] ** 2).sum() ** 0.5
                diff_from_to = (
                    (np.asarray(list(base_pose)) - center_loc) ** 2
                ).sum() ** 0.5

                if eye_depth - diff_from_to + 1e-2 >= 0:
                    can_clear.append(voxel)
                else:
                    block_obj = segment[
                        im_y, im_x
                    ]  # 0 is floor, 1 is table [the return value of maskrcnn]
                    if block_obj not in object_list.keys():  # TODO adhoc.
                        # if block_obj!=0:
                        print(f"ERROR: Sight blocked by an unknown object {block_obj}")
                        # else:
                        # can_clear.append(voxel) # TODO adhoc. ignore voxels blocked by the robot
                    else:
                        ent = object_list[block_obj]
                        if ent not in block_list.keys():
                            block_list[ent] = [[ent.loc, self.poses[voxel]]]
                        else:
                            block_list[ent].append([ent.loc, self.poses[voxel]])

        return can_clear, block_list

    def __repr__(self):
        return "vs{}".format(id(self) % 1000)


""" ========================================================================= """


class BeliefTask(object):
    def __init__(
        self,
        robot,
        arms=tuple(),
        grasp_types=tuple(),
        class_from_body={},
        sinks=tuple(),
        stoves=tuple(),
        movable=tuple(),
        surfaces=tuple(),
        rooms=tuple(),
        goal_localized=tuple(),
        goal_registered=tuple(),
        goal_holding=tuple(),
        goal_on=tuple(),
        goal_fo=tuple(),
    ):
        self.robot = robot
        self.arms = arms
        self.grasp_types = grasp_types
        self.class_from_body = class_from_body
        self.movable = movable
        self.surfaces = surfaces
        self.rooms = rooms
        self.goal_holding = goal_holding
        self.goal_on = goal_on
        self.goal_localized = goal_localized
        self.goal_registered = goal_registered
        self.goal_fo = goal_fo
        self.gripper = None
        self.sinks = sinks
        self.stoves = stoves

    def get_bodies(self):
        return self.movable + self.surfaces + self.rooms

    @property
    def fixed(self):
        movable = [self.robot] + list(self.movable)
        if self.gripper is not None:
            movable.append(self.gripper)
        return list(filter(lambda b: b not in movable, get_bodies()))

    def get_supports(self, body):
        if body in self.movable:
            return self.surfaces
        if body in self.surfaces:
            return self.rooms
        if body in self.rooms:
            return None
        raise ValueError(body)

    def get_gripper(self, arm="left"):
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, arm=arm)
        return self.gripper

    # def get_vspaces(self): # TODO multiple vspace
    #     return


# TODO: operate on histories to do open-world
class BeliefState(State):
    def __init__(
        self, task, b_on={}, registered={}, target=tuple(), block_list={}, **kwargs
    ):
        super(BeliefState, self).__init__(**kwargs)
        self.task = task
        self.b_on = b_on
        # self.localized = set(localized)
        self.registered = registered  # dict{obj_id_in_pybullet: Entity}
        self.block_list = block_list
        self.target = set(target)
        # TODO: store configurations
        """
        for body in task.get_bodies():
            if not self.is_localized(body):
                #self.poses[body] = None
                self.poses[body] = object()
                #del self.poses[body]
            #elif body not in registered:
            #    point, quat = self.poses[body].value
            #    self.poses[body] = Pose(body, (point, None))
        """

    def is_localized(self, body):
        for surface in self.b_on[body].support():
            if (surface != OTHER) and (LOCALIZED_PROB <= self.b_on[body].prob(surface)):
                return True
        return False

    def __repr__(self):
        items = []
        for b in sorted(self.b_on.keys()):
            d = self.b_on[b]
            support_items = [
                "{}: {:.2f}".format(s, d.prob(s)) for s in sorted(d.support(), key=str)
            ]
            items.append("{}: {{{}}}".format(b, ", ".join(support_items)))
        return "{}({},{})".format(
            self.__class__.__name__,
            # self.b_on,
            "{{{}}}".format(", ".join(items)),
            list(map(get_name, self.registered)),
        )


#######################################################


def set_uniform_belief(task, b_on, body, p_other=0.0):
    # p_other is the probability that it doesn't actually exist
    # TODO: option to bias towards particular bottom
    other = DeltaDist(OTHER)
    uniform = UniformDist(task.get_supports(body))
    b_on[body] = MixtureDD(other, uniform, p_other)


def set_delta_belief(task, b_on, body):
    supports = task.get_supports(body)
    if supports is None:
        b_on[body] = DeltaDist(supports)
        return
    for bottom in task.get_supports(body):
        if is_center_stable(body, bottom):
            b_on[body] = DeltaDist(bottom)
            return
    raise RuntimeError("No support for body {}".format(body))


#######################################################


def get_localized_rooms(task, **kwargs):
    # TODO: I support that in a closed world, it would automatically know where they are
    # TODO: difference between knowing position confidently and where it is
    b_on = {}
    for body in task.surfaces + task.movable:
        set_uniform_belief(task, b_on, body, **kwargs)
    for body in task.rooms:
        set_delta_belief(task, b_on, body)
    return BeliefState(task, b_on=b_on)


def get_localized_surfaces(task, **kwargs):
    b_on = {}
    for body in task.movable:
        set_uniform_belief(task, b_on, body, **kwargs)
    for body in task.rooms + task.surfaces:
        set_delta_belief(task, b_on, body)
    return BeliefState(task, b_on=b_on)


def get_localized_movable(task):
    b_on = {}
    for body in task.rooms + task.surfaces + task.movable:
        set_delta_belief(task, b_on, body)
    return BeliefState(task, b_on=b_on)


#######################################################
class VisTask(object):
    def __init__(
        self,
        robot,
        arms=tuple(),
        grasp_types=tuple(),
        class_from_body={},
        sinks=tuple(),
        stoves=tuple(),
        movable=tuple(),
        surfaces=tuple(),
        rooms=tuple(),
        # entities={},
        goal_localized=tuple(),
        goal_registered=tuple(),
        goal_holding=tuple(),
        goal_on=tuple(),
        goal_fo=tuple(),
        goal_category=3,
        vspace=tuple(),
    ):
        self.robot = robot
        self.arms = arms
        self.grasp_types = grasp_types
        self.class_from_body = class_from_body
        self.movable = movable
        self.surfaces = surfaces
        self.rooms = rooms
        self.vspace = vspace
        self.goal_holding = goal_holding
        self.goal_on = goal_on
        self.goal_localized = goal_localized
        self.goal_registered = goal_registered
        self.goal_fo = goal_fo
        self.gripper = None
        self.sinks = sinks
        self.stoves = stoves
        self.state = None
        # self.entities = entities
        self.goal_category = goal_category

    def get_bodies(self):
        return self.movable + self.surfaces + self.rooms

    @property
    def fixed(self):
        movable = [self.robot] + list(self.movable)
        if self.gripper is not None:
            movable.append(self.gripper)
        return list(filter(lambda b: b not in movable, get_bodies()))

    def get_supports(self, body):
        if body in self.movable:
            return self.surfaces
        if body in self.surfaces:
            return self.rooms
        if body in self.rooms:
            return None
        raise ValueError(body)

    def get_gripper(self, arm="left"):
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, arm=arm)
        return self.gripper

    # def get_vspaces(self): # TODO multiple vspace
    #     return
    def goal_success(self, entity):
        # print(f'entity category is {entity.category}, goal category is {self.goal_category}')
        if entity.category == self.goal_category:
            return True
        else:
            return False


#######################################################


def get_kitchen_task(arm="left", grasp_type="top", camonly=False):
    if not camonly:  # if CAMONLY==True, don't create the robot
        with HideOutput():
            pr2 = create_pr2(use_drake=USE_DRAKE_PR2)
        set_arm_conf(pr2, arm, get_carry_conf(arm, grasp_type))
        open_arm(pr2, arm)
        other_arm = get_other_arm(arm)
        set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
        close_arm(pr2, other_arm)

    # table, cabbage_list, sink, stove = create_random_kitchen()
    floor = get_bodies()[1]
    class_from_body = {
        table: "table",
        # cabbage: 'cabbage',
        # sink: 'sink',
        # stove: 'stove',
    }  # TODO: use for debug
    movable = []  # cabbage_list #[cabbage]
    surfaces = [table]
    if sink is not None:
        surfaces.append(sink)
    if stove is not None:
        surfaces.append(stove)
    rooms = [floor]

    # TODO observation test
    camera_pos = (0.15, 0, 1.37)
    if camonly:
        grid = Voxelgrid(0.03, (0.5, -0.5, 0.7), (1.0, 0.5, 1.4))
    else:
        grid = Voxelgrid(0.1, (0.5, -0.5, 0.7), (1.0, 0.5, 1.4))

    # task.goal_fo = [grid]
    # print ('task.goal_fo is ',task.goal_fo)

    if not camonly:
        return VisTask(
            robot=pr2,
            arms=[arm],
            grasp_types=[grasp_type],
            class_from_body=class_from_body,
            movable=movable,
            surfaces=surfaces,
            rooms=rooms,
            vspace=grid,
            # sinks=[sink], stoves=[stove],
            # goal_localized=[cabbage],
            # goal_registered=[cabbage],
            # goal_holding=[(arm, cabbage_list[0])],
            # goal_on=[(cabbage, table)],
            # goal_on=[(cabbage, stove) for cabbage in cabbage_list],
            # goal_fo = [grid]
        )
    else:
        return rooms, grid


def get_problem1(localized="rooms", camonly=False, **kwargs):
    if camonly:
        room, vspace = get_kitchen_task(camonly=camonly)
        return room, vspace
    else:
        task = get_kitchen_task(camonly=camonly)
    if localized == "rooms":
        initial = get_localized_rooms(task, **kwargs)
    elif localized == "surfaces":
        initial = get_localized_surfaces(task, **kwargs)
    elif localized == "movable":
        initial = get_localized_movable(task)
    else:
        raise ValueError(localized)
    task.state = initial
    return task, initial
