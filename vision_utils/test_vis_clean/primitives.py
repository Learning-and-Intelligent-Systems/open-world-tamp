from __future__ import print_function

import numpy as np
from examples.discrete_belief.dist import DDist
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (  # , Entity
    SELF_COLLISIONS, Attach, Command, Conf, Detach, Pose, Trajectory,
    create_trajectory, get_target_path)
from examples.pybullet.utils.pybullet_tools.pr2_problems import \
    get_fixed_bodies
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    HEAD_LINK_NAME, MAX_KINECT_DISTANCE, attach_viewcone, get_detection_cone,
    get_group_conf, get_group_joints, get_kinect_registrations, get_viewcone,
    get_visual_detections, inverse_visibility, plan_scan_path, set_group_conf,
    visible_base_generator)
from examples.pybullet.utils.pybullet_tools.utils import (
    CLIENT, GREEN, INF, PI, RED, BodySaver, CameraImage, LockRenderer,
    add_text, apply_alpha, child_link_from_joint, create_cylinder, create_mesh,
    dump_body, get_body_name, get_length, get_link_name, get_link_pose,
    get_link_subtree, get_name, get_projection_matrix, is_center_stable,
    link_from_name, multiply, pairwise_collision, plan_direct_joint_motion,
    plan_waypoints_joint_motion, point_from_pose, remove_body, set_color,
    set_euler, set_joint_positions, set_point, set_pose, unit_pose,
    wait_for_duration, wait_for_user)

VIS_RANGE = (0.5, 1.5)
REG_RANGE = (0.5, 1.5)
ROOM_SCAN_TILT = np.pi / 6
P_LOOK_FP = 0
P_LOOK_FN = 0  # 1e-1
import pybullet as p


def get_image(
    camera_pos,
    target_pos,
    width=640,
    height=480,
    vertical_fov=PI / 3,
    near=0.01,
    far=1000.0,
    segment=False,
    segment_links=False,
    client=CLIENT,
    lightspc=1.0,
):  # modified by xiaolin Dec.23, 2019. original version is get_image_robotview
    diff = np.asarray(list(target_pos)) - np.asarray(list(camera_pos))
    diff_len = (diff**2).sum() ** 0.5

    right = np.cross(diff, np.asarray([0, 0, 1]))
    up_vector = np.cross(right, diff)
    if (up_vector**2).sum() == 0:  # look perpendicular to the ground
        up_vector = np.asarray([1, 0, 0])

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vector,
        physicsClientId=client,
    )
    projection_matrix = get_projection_matrix(width, height, vertical_fov, near, far)

    if segment:
        if segment_links:
            flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        else:
            flags = 0
    else:
        flags = p.ER_NO_SEGMENTATION_MASK
    image = CameraImage(
        *p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            shadow=False,
            flags=flags,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,  # p.ER_TINY_RENDERER, # p.ER_BULLET_HARDWARE_OPENGL
            physicsClientId=client,
            lightSpecularCoeff=lightspc,
        )[2:]
    )
    depth = image.depthPixels
    segmented = image.segmentationMaskBuffer
    return (
        CameraImage(image.rgbPixels, depth, segmented),
        view_matrix,
        projection_matrix,
        width,
        height,
    )


def get_isTarget_test(task):
    def gen(entity):
        print(f"entity is {entity}, category {entity.category}")
        if task.goal_success(entity):
            return True
        else:
            return

    return gen


def get_isGraspable_test(task):
    def gen(entity):
        if entity.obj_id != -1 and entity.graspable:
            return True
        else:
            return

    return gen


def get_entity_pose(task):
    def gen(entity):
        while True:
            if entity.obj_id == -1:
                return
            else:
                yield (entity.loc,)
                return

    return gen


def get_in_range_test(task, range):
    # TODO: could just test different visibility w/o ranges
    def test(o, p, bq):
        if o in task.rooms:
            return True
        target_xy = point_from_pose(p.value)[:2]
        base_xy = bq.values[:2]
        return range[0] <= get_length(np.array(target_xy) - base_xy) <= range[1]

    return test


def get_fo_test(task):
    def test(vs):
        if vs.fully_observed():
            return True
        else:
            return

    return test


def get_vis_base_gen(task, base_range, collisions=False):
    robot = task.robot
    base_joints = get_group_joints(robot, "base")
    obstacles = get_fixed_bodies(task) if collisions else []

    def gen(o, p):
        if o in task.rooms:  # TODO: predicate instead
            return
        # default_conf = arm_conf(a, g.carry)
        # joints = get_arm_joints(robot, a)
        # TODO: check collisions with fixed links
        target_point = point_from_pose(p.value)
        base_generator = visible_base_generator(robot, target_point, base_range)
        while True:
            set_pose(o, p.value)  # p.assign()
            bq = Conf(robot, base_joints, next(base_generator))
            # bq = Pose(robot, get_pose(robot))
            bq.assign()
            if any(pairwise_collision(robot, b) for b in obstacles):
                yield None
            else:
                yield (bq,)

    # TODO: return list_fn & accelerated
    return gen


def get_inverse_visibility_fn(task, collisions=True):
    robot = task.robot
    head_joints = get_group_joints(robot, "head")
    obstacles = get_fixed_bodies(task) if collisions else []

    def fn(o, p, bq):
        set_pose(o, p.value)  # p.assign()
        bq.assign()
        if o in task.rooms:
            waypoints = plan_scan_path(task.robot, tilt=ROOM_SCAN_TILT)
            set_group_conf(robot, "head", waypoints[0])
            path = plan_waypoints_joint_motion(
                robot,
                head_joints,
                waypoints[1:],
                obstacles=obstacles,
                self_collisions=SELF_COLLISIONS,
            )
            if path is None:
                return None
            ht = create_trajectory(robot, head_joints, path)
            hq = ht.path[0]
        else:
            target_point = point_from_pose(p.value)
            head_conf = inverse_visibility(robot, target_point)
            if head_conf is None:  # TODO: test if visible
                return None
            hq = Conf(robot, head_joints, head_conf)
            ht = Trajectory([hq])
        return (hq, ht)

    return fn


def get_inverse_visibility_fixbase_fn(task, collisions=True):
    robot = task.robot
    head_joints = get_group_joints(robot, "head")
    obstacles = get_fixed_bodies(task) if collisions else []

    def fn(p, vs, bq=None):
        # TODO collision check (vis blocked or not)
        if bq is not None:
            bq.assign()
        # if o in task.rooms:
        #     waypoints = plan_scan_path(task.robot, tilt=ROOM_SCAN_TILT)
        #     set_group_conf(robot, 'head', waypoints[0])
        #     path = plan_waypoints_joint_motion(robot, head_joints, waypoints[1:],
        #                                   obstacles=obstacles, self_collisions=SELF_COLLISIONS)
        #     if path is None:
        #         return None
        #     ht = create_trajectory(robot, head_joints, path)
        #     hq = ht.path[0]
        # else:
        target_point = point_from_pose(p.value)

        head_conf = inverse_visibility(robot, target_point)
        if head_conf is None:  # TODO: test if visible
            return None

        # import ipdb
        # ipdb.set_trace()
        hq = Conf(robot, head_joints, head_conf)
        ht = Trajectory([hq])
        return (hq, ht)
        # return (hq, ht)

    return fn


def get_visclear_test(task):
    obstacles = task.get_bodies()
    obstacles.extend([o for o in task.state.registered.keys()])
    vspace = task.vspace

    def fn(pvis):
        base_pose = (0.15, 0, 1.37)  # TODO adhoc
        target_pose = pvis.value[0]
        diff = np.asarray(list(target_pose)) - np.asarray(list(base_pose))
        diff_len = (diff**2).sum() ** 0.5
        # tilt = np.arcsin(abs(diff[2])/diff_len)
        # pan = np.arctan(diff[1]/diff[0])
        # (rgba, depth, segment), view_matrix, projection_matrix, _, _  = get_image(base_pose, pan,
        #         tilt,segment=False)
        (rgba, depth, segment), view_matrix, projection_matrix, _, _ = get_image(
            base_pose, target_pose, segment=False
        )
        depth = (depth[240, 320] - 0.5) * 2  # scale from 0-1 to -1~1
        normed_loc = np.asarray([0, 0, depth, 1])
        # loc in normalized space(-1~1)
        eye_loc = np.linalg.inv(
            np.asarray(projection_matrix).reshape(4, 4).transpose(1, 0)
        ).dot(normed_loc)

        actual_depth = -eye_loc[2] / eye_loc[3]
        # depth = (far * near / (far - (far - near) * depth))[240,320]
        if (
            actual_depth - diff_len + 1e-2 >= 0
        ):  # add 1e-2 to account for OpenGL potential numerical error
            return True
        else:
            print(target_pose, "actual depth: ", actual_depth, "diff_len_sim", diff_len)
            return
        # collide=False
        # camera_pos=(0.15,0,1.37) #TODO adhoc
        # target_point = point_from_pose(pvis.value)
        # diff = np.asarray(list(target_point))-np.asarray(list(camera_pos))
        # diff_len = (diff**2).sum()**0.5
        # tilt = np.arcsin(-diff[2]/diff_len)
        # pan = np.arctan(diff[1]/diff[0])
        # vis_ray = create_cylinder(0.01, diff_len)
        # center = np.divide(np.add(target_point,camera_pos),2.)
        # set_point(vis_ray,center)
        # set_euler(vis_ray,(0,np.pi/2.+tilt,pan))
        # for obstacle in obstacles:
        #     if pairwise_collision(obstacle,vis_ray):
        #         collide=True
        #         break
        # remove_body(vis_ray)
        # if collide:
        #     return
        # else:
        #     return True

    return fn


def get_unblock_test(task, collisions=True):
    robot = task.robot

    def fn(
        o, p, vs
    ):  # TODO adhoc.TEST it's invalid if block anything. should be sampled from observed area
        base_pose = (0.15, 0, 1.37)  # TODO adhoc
        target_pose = p.value[0]
        voxel = task.vspace.voxel_from_point(target_pose)
        if (
            tuple(voxel) not in task.vspace.occupied
        ):  # if the voxel has been observed, considered as valid
            return True
        else:
            return
        # diff = np.asarray(list(target_pose)) - np.asarray(list(base_pose))
        # diff_len = (diff**2).sum()**0.5
        # (rgba, depth, segment), view_matrix, projection_matrix, _, _  =
        # get_image(base_pose, target_pose,segment=False) # can't use visual sim to decide it because sight is occluded by robot arm
        # depth = (depth[240,320] - 0.5) * 2 # scale from 0-1 to -1~1
        # normed_loc = np.asarray([0,0,depth,1])
        # # loc in normalized space(-1~1)
        # eye_loc = np.linalg.inv(np.asarray(projection_matrix).reshape(4,4).transpose(1,0)).dot(normed_loc)
        # actual_depth = -eye_loc[2]/eye_loc[3]
        # if actual_depth-diff_len+1e-2>=0:#add 1e-2 to account for OpenGL potential numerical error
        #     return True
        # else:
        #     return

        # if isinstance(o,Entity):#TODO adhoc
        #     o = o.obj_id
        # p.assign()
        # obstacles = list(vs.occupied)
        # if len(obstacles)==0:
        #     return True
        # box=vs.create_box(vs.center_from_voxel(obstacles[0]))
        # collide=False
        # for obstacle in obstacles:
        #     set_point(box,vs.center_from_voxel(obstacle))
        #     if pairwise_collision(o,box):
        #         collide=True
        #         break
        # remove_body(box)
        # if collide:
        #     return
        # else:
        #     return True

        # if isinstance(o,Entity):
        #     o = o.obj_id
        # camera_pos=(0.15,0,1.37) #TODO adhoc
        # target_point = point_from_pose(pvis.value)
        # set_pose(o,p.value)
        # diff = np.asarray(list(target_point))-np.asarray(list(camera_pos))
        # diff_len = (diff**2).sum()**0.5
        # tilt = np.arcsin(-diff[2]/diff_len)
        # pan = np.arctan(diff[1]/diff[0])
        # vis_ray = create_cylinder(0.01, diff_len)
        # center = np.divide(np.add(target_point,camera_pos),2.)
        # set_point(vis_ray,center)
        # set_euler(vis_ray,(0,np.pi/2.+tilt,pan))
        # if not pairwise_collision(o,vis_ray):
        #     remove_body(vis_ray)
        #     return True
        # else:
        #     remove_body(vis_ray)
        #     return

    return fn


#######################################################


def plan_head_traj(task, head_conf):
    robot = task.robot
    obstacles = get_fixed_bodies(task)  # TODO: movable objects
    # head_conf = get_joint_positions(robot, head_joints)
    # head_path = [head_conf, hq.values]
    head_joints = get_group_joints(robot, "head")
    head_path = plan_direct_joint_motion(
        robot,
        head_joints,
        head_conf,
        obstacles=obstacles,
        self_collisions=SELF_COLLISIONS,
    )
    assert head_path is not None
    return create_trajectory(robot, head_joints, head_path)


def inspect_trajectory(task, trajectory):
    if not trajectory.path:
        return
    robot = trajectory.path[0].body
    obstacles = get_fixed_bodies(task)  # TODO: movable objects
    # TODO: minimum distance of some sort (to prevent from looking at the bottom)
    # TODO: custom lower limit as well
    head_waypoints = []
    for target_point in get_target_path(trajectory):
        head_conf = inverse_visibility(robot, target_point)
        # TODO: could also draw the sequence of inspected points as the head moves
        if head_conf is None:
            continue
        head_waypoints.append(head_conf)
    head_joints = get_group_joints(robot, "head")
    # return create_trajectory(robot, head_joints, head_waypoints)
    head_path = plan_waypoints_joint_motion(
        robot,
        head_joints,
        head_waypoints,
        obstacles=obstacles,
        self_collisions=SELF_COLLISIONS,
    )
    assert head_path is not None
    return create_trajectory(robot, head_joints, head_path)


def move_look_trajectory(task, trajectory, max_tilt=np.pi / 6):  # max_tilt=INF):
    # TODO: implement a minimum distance instead of max_tilt
    # TODO: pr2 movement restrictions
    # base_path = [pose.to_base_conf() for pose in trajectory.path]
    base_path = trajectory.path
    if not base_path:
        return trajectory
    obstacles = get_fixed_bodies(task)  # TODO: movable objects
    robot = base_path[0].body
    target_path = get_target_path(trajectory)
    waypoints = []
    index = 0
    with BodySaver(robot):
        # current_conf = base_values_from_pose(get_pose(robot))
        for i, conf in enumerate(base_path):  # TODO: just do two loops?
            conf.assign()
            while index < len(target_path):
                if i < index:
                    # Don't look at past or current conf
                    target_point = target_path[index]
                    target_point = target_path[index]
                    head_conf = inverse_visibility(
                        robot, target_point
                    )  # TODO: this is slightly slow
                    # print(index, target_point, head_conf)
                    if (head_conf is not None) and (head_conf[1] < max_tilt):
                        break
                index += 1
            else:
                head_conf = get_group_conf(robot, "head")
            set_group_conf(robot, "head", head_conf)
            # print(i, index, conf.values, head_conf) #, get_pose(robot))
            waypoints.append(np.concatenate([conf.values, head_conf]))
    joints = tuple(base_path[0].joints) + tuple(get_group_joints(robot, "head"))
    # joints = get_group_joints(robot, 'base') + get_group_joints(robot, 'head')
    # set_pose(robot, unit_pose())
    # set_group_conf(robot, 'base', current_conf)
    path = plan_waypoints_joint_motion(
        robot, joints, waypoints, obstacles=obstacles, self_collisions=SELF_COLLISIONS
    )
    return create_trajectory(robot, joints, path)
    # Pose(robot, pose_from_base_values(q, bq1.value))
    # new_traj.path.append(Pose(...))


#######################################################


class AttachCone(Command):  # TODO: make extend Attach?
    def __init__(self, robot):
        self.robot = robot
        self.group = "head"
        self.cone = None

    def apply(self, state, **kwargs):
        with LockRenderer():
            self.cone = get_viewcone(color=apply_alpha(RED, 0.5))
            state.poses[self.cone] = None
            cone_pose = Pose(self.cone, unit_pose())
            attach = Attach(self.robot, self.group, cone_pose, self.cone)
            attach.assign()
            wait_for_duration(1e-2)
        for _ in attach.apply(state, **kwargs):
            yield

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class DetachCone(Command):  # TODO: make extend Detach?
    def __init__(self, attach):
        self.attach = attach

    def apply(self, state, **kwargs):
        cone = self.attach.cone
        detach = Detach(self.attach.robot, self.attach.group, cone)
        for _ in detach.apply(state, **kwargs):
            yield
        del state.poses[cone]
        remove_body(cone)
        wait_for_duration(1e-2)

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def get_cone_commands(robot):
    attach = AttachCone(robot)
    detach = DetachCone(attach)
    return attach, detach


#######################################################


def get_observation_fn(surface, p_look_fp=P_LOOK_FP, p_look_fn=P_LOOK_FN):
    # TODO: clip probabilities so doesn't become zero
    def fn(s):
        # P(obs | s1=loc1, a=control_loc)
        if s == surface:
            return DDist({True: 1 - p_look_fn, False: p_look_fn})
        return DDist({True: p_look_fp, False: 1 - p_look_fp})

    return fn


# TODO: update whether localized on scene


class Scan(Command):
    _duration = 0.5

    def __init__(self, robot, surface, detect=True, camera_frame=HEAD_LINK_NAME):
        self.robot = robot
        self.surface = surface
        self.camera_frame = camera_frame
        self.link = link_from_name(robot, self.camera_frame)
        self.detect = detect

    def apply(self, state, **kwargs):
        # TODO: identify surface automatically
        with LockRenderer():
            cone = get_viewcone(color=apply_alpha(RED, 0.5))
            set_pose(cone, get_link_pose(self.robot, self.link))
            wait_for_duration(1e-2)
        wait_for_duration(self._duration)  # TODO: don't sleep if no viewer?
        remove_body(cone)
        wait_for_duration(1e-2)

        if self.detect:
            # TODO: the collision geometries are being visualized
            # TODO: free the renderer
            head_joints = get_group_joints(self.robot, "head")
            exclude_links = set(
                get_link_subtree(self.robot, child_link_from_joint(head_joints[-1]))
            )
            detections = get_visual_detections(
                self.robot,
                camera_link=self.camera_frame,
                exclude_links=exclude_links,
            )
            # color=apply_alpha(RED, 0.5))
            print("Detections:", detections)
            for body, dist in state.b_on.items():
                obs = (body in detections) and (is_center_stable(body, self.surface))
                if obs or (self.surface not in state.task.rooms):
                    # TODO: make a command for scanning a room instead?
                    dist.obsUpdate(get_observation_fn(self.surface), obs)
            # state.localized.update(detections)
        # TODO: pose for each object that can be real or fake
        yield

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, get_body_name(self.surface))


class ScanRoom(Command):
    _tilt = np.pi / 6

    def __init__(self, robot, surface):
        self.robot = robot
        self.surface = surface

    def apply(self, state, **kwargs):
        assert self.surface in state.task.rooms
        obs_fn = get_observation_fn(self.surface)
        for body, dist in state.b_on.items():
            if 0 < dist.prob(self.surface):
                dist.obsUpdate(obs_fn, True)
        # detections = get_visual_detections(self.robot)
        # print('Detections:', detections)
        # for body, dist in state.b_on.items():
        #    obs = (body in detections) and (is_center_stable(body, self.surface))
        #    dist.obsUpdate(obs_fn, obs)
        yield

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, get_body_name(self.surface))


#######################################################


class Detect(Command):
    def __init__(self, robot, surface, body):
        self.robot = robot
        self.surface = surface
        self.body = body

    def apply(self, state, **kwargs):
        yield
        # TODO: need to be careful that we don't move in between this...
        # detections = get_visual_detections(self.robot)
        # print('Detections:', detections)
        # dist = state.b_on[self.body]
        # obs = (self.body in detections)
        # dist.obsUpdate(get_observation_fn(self.surface), obs)
        # if obs:
        #     state.localized.add(self.body)
        # TODO: pose for each object that can be real or fake

    def __repr__(self):
        return "{}({},{})".format(
            self.__class__.__name__, get_name(self.surface), get_name(self.body)
        )


class Mark(Command):
    def __init__(self, robot, body):
        self.robot = robot
        self.body = body

    def apply(self, state, **kwargs):
        state.target.add(self.body)
        add_text("target obj", position=self.body.loc.value[0], parent=self.body.obj_id)
        yield
        # TODO: need to be careful that we don't move in between this...
        # detections = get_visual_detections(self.robot)
        # print('Detections:', detections)
        # dist = state.b_on[self.body]
        # obs = (self.body in detections)
        # dist.obsUpdate(get_observation_fn(self.surface), obs)
        # if obs:
        #     state.localized.add(self.body)
        # TODO: pose for each object that can be real or fake

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


import matplotlib.pyplot as plt


class Observe(Command):
    def __init__(self, pose, base_pose, vspace):
        self.vspace = vspace
        self.pose = pose
        self.base_pose = base_pose
        self.object_list = None

    def apply(self, state, **kwargs):
        base_pose = (0.15, 0, 1.37)  # TODO
        pose = self.pose
        vspace = self.vspace

        target_pose = pose.value[0]
        # diff = np.asarray(list(target_pose)) - np.asarray(list(base_pose))
        # #b = diff[0] / (diff[0]**2+diff[2]**2)**0.5
        # #a = (1-b*b)**0.5
        # #upvector = [a,0,b]
        # #view_matrix = p.computeViewMatrix(cameraEyePosition=base_pose,
        # #        cameraTargetPosition=target_pose,cameraUpVector=upvector)
        # #projection_matrix = get_projection_matrix
        # diff_len = (diff**2).sum()**0.5
        # tilt = np.arcsin(abs(diff[2])/diff_len)
        # pan = np.arctan(diff[1]/diff[0])

        # vspace.hide_vblock()

        (rgba, depth, segment), view_matrix, projection_matrix, _, _ = get_image(
            base_pose, target_pose, segment=True
        )
        #         (rgba, depth, segment), view_matrix, projection_matrix, _, _  = get_image(base_pose, pan,
        #                 tilt,segment=True)

        debug = False
        if debug:
            plt.imsave(f"render/rgba_{pose}.png", rgba)
            plt.imsave(f"render/dep_{pose}.png", depth)
            print(f"save image to {pose}")
            import pdb

            pdb.set_trace()
        # plt.imsave(f'render/seg_{pose}.png',segment)
        # vspace.show_vblock()
        # if 6 in segment:
        #     import pdb
        #     pdb.set_trace()
        # pan = 0. #np.random.uniform(-np.pi/2,np.pi/2)
        # tilt = 0. #np.random.uniform(np.pi/6,np.pi/3)
        # # (rgba, depth, segment), view_matrix, projection_matrix, width, height = get_image(base_pose, pan, tilt,segment=False)
        # # plt.imsave(f'rgba_{self.pose}.png',rgba)

        object_list = np.unique(segment)
        n_object = object_list.shape[0]
        entity_list = []
        if (
            object_list.max() >= 3
        ):  # TODO adhoc. -1 Undefined, 0 plane, 1robot, 2 desk, 6 don'tknow
            for obj_i in object_list:
                if obj_i not in state.registered.keys() and obj_i >= 3:  # and obj_i!=6:
                    # state.task.entities+=(obj_i,)
                    new_ent = Entity(obj_i, obj_i)
                    state.registered[obj_i] = new_ent
                    entity_list.append(new_ent)
                elif obj_i in state.registered.keys() and obj_i >= 3:  # and obj_i!=6:
                    entity_list.append(state.registered[obj_i])
        ########################################################################
        # TODO replace the pybulelt collision check with RGBD image processing
        ########################################################################

        # width = 640
        # height = 480
        # projection_matrix = get_projection_matrix(width, height, 70.0, 0.02, 5.0) # See pybullet_tools/utils L770.

        # projection_matrix = np.asarray(projection_matrix).reshape(4,4)
        # near = (projection_matrix[0,0] * width / 2. + projection_matrix[1,1] * height / 2.) /2.
        # far = ((projection_matrix[2,2]-1.)/(projection_matrix[2,2]+1.) * near + (projection_matrix[3,2] * near) / (projection_matrix[3,2] + 2 * near)) / 2.
        # far_plane = []
        # far_plane.append(np.array([far, far*width/2./near,
        #     far*height/2./near])/30000.)
        # far_plane.append(np.array([far, far*width/2./near,
        #     -far*height/2./near])/30000.)
        # far_plane.append(np.array([far, -far*width/2./near,
        #     -far*height/2./near])/30000.)
        # far_plane.append(np.array([far, -far*width/2./near,
        #     far*height/2./near])/30000.)

        # viewcone = get_viewcone_proj(base_pose, far_plane)
        # set_point(viewcone, base_pose)
        # set_euler(viewcone, (0,tilt,pan))

        # affected = vspace.get_coarse_affected(viewcone)
        # block_list = vspace.check_collision(viewcone,affected,camera_pos=base_pose,occlusion=True,state=state)
        can_free_voxel, block_list = vspace.check_vis(
            target_pose, entity_list=entity_list
        )
        print("Block list:", block_list)
        for block_obj in block_list.keys():
            if block_obj not in state.block_list.keys():
                state.block_list[block_obj] = block_list[block_obj]
            else:
                for block_value in block_list[block_obj]:
                    if block_value not in state.block_list[block_obj]:
                        state.block_list[block_obj].append(block_value)

        # remove_body(viewcone)

        state.task.vspace.set_frees(can_free_voxel)
        self.object_list = object_list
        print(
            f"After command: len unobserved {len(self.vspace.occupied)}, object list {object_list}"
        )
        vislist = []
        for voxel in vspace.occupied:
            vislist.append(
                vspace.create_box(vspace.center_from_voxel(voxel), color=(1, 0, 0, 0.3))
            )
        wait_for_duration(3)
        for tmpbox in vislist:
            remove_body(tmpbox)

        yield

    def __repr__(self):
        return "{}({},{},{})".format(
            self.__class__.__name__,
            self.vspace,
            self.object_list,
            len(self.vspace.occupied),
        )


class Register(Command):
    _duration = 1.0

    def __init__(
        self, robot, body, camera_frame=HEAD_LINK_NAME, max_depth=MAX_KINECT_DISTANCE
    ):
        self.robot = robot
        self.body = body
        self.camera_frame = camera_frame
        self.max_depth = max_depth
        self.link = link_from_name(robot, camera_frame)

    def control(self, **kwargs):
        # TODO: filter for target object and location?
        return get_kinect_registrations(self.robot)

    def apply(self, state, **kwargs):
        # TODO: check if actually can register
        mesh, _ = get_detection_cone(
            self.robot, self.body, camera_link=self.camera_frame, depth=self.max_depth
        )
        if mesh is None:
            wait_for_user()
        assert mesh is not None
        with LockRenderer():
            cone = create_mesh(mesh, color=apply_alpha(GREEN, 0.5))
            wait_for_duration(1e-2)
        wait_for_duration(self._duration)
        remove_body(cone)
        wait_for_duration(1e-2)
        state.registered.add(self.body)
        yield

    def __repr__(self):
        return "{}({},{})".format(
            self.__class__.__name__, get_body_name(self.robot), get_name(self.body)
        )
