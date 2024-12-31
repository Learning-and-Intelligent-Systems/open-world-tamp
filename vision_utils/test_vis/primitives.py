from __future__ import print_function

import cv2
import numpy as np
from examples.discrete_belief.dist import DDist
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (  # , Entity
    SELF_COLLISIONS, Attach, Command, Conf, Detach, Pose, Trajectory,
    create_trajectory, get_target_path)
from examples.pybullet.utils.pybullet_tools.pr2_problems import \
    get_fixed_bodies
from examples.pybullet.utils.pybullet_tools.pr2_utils import (  # , get_viewcone_proj
    HEAD_LINK_NAME, MAX_KINECT_DISTANCE, attach_viewcone, get_detection_cone,
    get_group_conf, get_group_joints, get_kinect_registrations, get_viewcone,
    get_visual_detections, inverse_visibility, plan_scan_path, set_group_conf,
    visible_base_generator)
from examples.pybullet.utils.pybullet_tools.utils import (
    CLIENT, GREEN, INF, PI, RED, BodySaver, CameraImage, LockRenderer,
    add_text, apply_alpha, child_link_from_joint, create_cylinder, create_mesh,
    dump_body, get_body_name, get_length, get_link_name, get_link_pose,
    get_link_subtree, get_name, get_projection_matrix, is_center_stable,
    link_from_name, load_pybullet, mesh_from_points, multiply,
    pairwise_collision, plan_direct_joint_motion, plan_waypoints_joint_motion,
    point_from_pose, read_obj, remove_body, set_color, set_euler,
    set_joint_positions, set_point, set_pose, unit_pose, vertices_from_rigid,
    wait_for_duration, wait_for_user)

from .constant import (OBS_IOU_THRESHOLD, OBS_TIME_THRESHOLD,
                       REAL_WORLD_CLIENT, YCB_BANK_DIR)

VIS_RANGE = (0.5, 1.5)
REG_RANGE = (0.5, 1.5)
ROOM_SCAN_TILT = np.pi / 6
P_LOOK_FP = 0
P_LOOK_FN = 0  # 1e-1


class Entity(object):
    def __init__(
        self,
        obj_id,
        obj_id_head,
        category,
        loc=((0, 0, 0), (0, 0, 0, 0)),
        model=None,
        pose_uncertain=True,
        iou=0.0,
        obs_time=1,
    ):
        self.obj_id = obj_id
        self.category = category
        self.obj_id_head = obj_id_head
        self.loc = Pose(obj_id, value=loc)
        self.graspable = True  # TODO adhoc. all set true for now
        self.model = model  # vertices # known CAD model or reconstructed model
        self.pose_uncertain = pose_uncertain
        self.iou = iou
        self.obs_time = obs_time
        # TODO 6d pose

    def update_pose(self, pose, fix_symbol=False):  # use the same symbol or not
        if isinstance(pose, Pose):
            self.loc = pose
            new_head_tran, new_head_quat = self.loc.value
            set_pose(
                self.obj_id_head,
                (
                    tuple(np.asarray(new_head_tran) + np.asarray([0, 2, 0])),
                    new_head_quat,
                ),
            )
            # pose.assign()
        else:
            if fix_symbol:
                self.loc.value = pose
            else:
                self.loc = Pose(self.obj_id, pose)
            new_head_tran, new_head_quat = self.loc.value
            set_pose(
                self.obj_id_head,
                (
                    tuple(np.asarray(new_head_tran) + np.asarray([0, 2, 0])),
                    new_head_quat,
                ),
            )

    def __repr__(self):
        return "entity_{}".format(id(self) % 1000)


def get_isTarget_test(task):
    def gen(entity):
        if task.goal_success(entity):
            return True
        else:
            return

    return gen


def get_isGraspable_test(task):
    def gen(entity):
        if entity.graspable:  # entity.obj_id!=-1 and
            return True  # TODO assume all ycb objects are graspable right now
        else:
            return

    return gen


def get_isPoseCertain_test(task):
    def gen(entity):
        if entity.pose_uncertain:
            return
        else:
            return True

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
        target_point = point_from_pose(p.value)

        head_conf = inverse_visibility(robot, target_point)
        if head_conf is None:  # TODO: test if visible
            return None

        hq = Conf(robot, head_joints, head_conf)
        ht = Trajectory([hq])
        return (hq, ht)

    return fn


def get_visclear_test(task):
    obstacles = task.get_bodies()
    obstacles.extend([o for o in task.state.registered.keys()])
    vspace = task.vspace

    def fn(pvis):
        base_pose = EYE_BASE_POSITION  # TODO adhoc
        target_pose = pvis.value[0]
        diff = np.asarray(list(target_pose)) - np.asarray(list(base_pose))
        diff_len = (diff**2).sum() ** 0.5
        (rgba, depth, segment), view_matrix, projection_matrix, _, _ = get_image(
            base_pose, target_pose, segment=False, client=REAL_WORLD_CLIENT
        )
        depth = (depth[240, 320] - 0.5) * 2  # scale from 0-1 to -1~1
        normed_loc = np.asarray([0, 0, depth, 1])
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

    return fn


def get_unblock_test(task, collisions=True):
    robot = task.robot

    def fn(
        o, p, vs
    ):  # TODO adhoc.TEST it's invalid if block anything. should be sampled from observed area
        # base_pose = EYE_BASE_POSITION#TODO adhoc
        target_pose = p.value[0]
        voxel = task.vspace.voxel_from_point(target_pose)
        if (
            tuple(voxel) not in task.vspace.occupied
        ):  # if the voxel has been observed, considered as valid
            return True
        else:
            return

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

# class AttachCone(Command): # TODO: make extend Attach?
#     def __init__(self, robot):
#         self.robot = robot
#         self.group = 'head'
#         self.cone = None
#     def apply(self, state, **kwargs):
#         with LockRenderer():
#             self.cone = get_viewcone(color=apply_alpha(RED, 0.5))
#             state.poses[self.cone] = None
#             cone_pose = Pose(self.cone, unit_pose())
#             attach = Attach(self.robot, self.group, cone_pose, self.cone)
#             attach.assign()
#             wait_for_duration(1e-2)
#         for _ in attach.apply(state, **kwargs):
#             yield
#     def __repr__(self):
#         return '{}()'.format(self.__class__.__name__)


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
        add_text(
            "target obj",
            position=(0, 0, 0.2),
            color=(1.0, 0, 0),
            parent=self.body.obj_id,
        )  # position is the relative position between text and obj
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


def get_bbox(idx, segment):
    # TODO adhoc
    border_list = [
        -1,
        40,
        80,
        120,
        160,
        200,
        240,
        280,
        320,
        360,
        400,
        440,
        480,
        520,
        560,
        600,
        640,
        680,
    ]
    img_width = 480
    img_length = 640

    py, px = np.where(segment == idx)
    rmin, rmax, cmin, cmax = np.min(py), np.max(py), np.min(px), np.max(px)
    # rmin = int(posecnn_rois[idx][3]) + 1
    # rmax = int(posecnn_rois[idx][5]) - 1
    # cmin = int(posecnn_rois[idx][2]) + 1
    # cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


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


import copy

import numpy.ma as ma
import torch
import torchvision.transforms as transforms


def get_pose6d(rgba, depth, xyz, segment, estimator, refiner):
    """
    input:
        rgba: HxWx4.
        depth: HxW. NOTE: This depth is only used for checking the holes, the
        cloud is still from xyz. some changes will be needed if want to applied
        on real images

        xyz: HxWx3. NOTE: The coordinate is different. The positive direction
        of z axis is inversed. Also, the coordinate is different with pybullet
        default coordinate, so be careful.

        segment: Hxw
    """

    # TODO adhoc
    num_points = 1000
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    bs = 1

    xyz[:, :, 2] *= -1
    xyz[:, :, 1] *= -1
    obj_list = np.unique(segment)
    pred_wo_refine = {}
    # estimator, refiner = estimator
    for idx, obj in enumerate(obj_list):
        if obj <= 1:  # or obj==2 or obj==-1: #skip floor and desk and robot
            continue
        rmin, rmax, cmin, cmax = get_bbox(obj, segment)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        # mask_label = ma.getmaskarray(ma.masked_equal(segment,obj)) # TODO
        mask = np.ones(depth.shape)  # mask_label*mask_depth
        # import pdb;pdb.set_trace()
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), "wrap")

        cloud = xyz[rmin:rmax, cmin:cmax, :].reshape(-1, 3)[choose].astype(np.float32)
        cloud = torch.as_tensor(cloud).cuda().view(1, num_points, 3)

        img_masked = rgba[:, :, :3].transpose(2, 0, 1)[:, rmin:rmax, cmin:cmax]
        img_masked = norm(
            torch.from_numpy(img_masked.astype(np.float32)) / 255.0
        )  # TODO /255.? 2 bugs:1.Target object is masked as dark. 2. over bright
        img_masked = torch.as_tensor(img_masked).cuda()
        img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

        choose = torch.LongTensor(choose.astype(np.int32)).cuda()
        index = torch.LongTensor(
            [obj - 2]
        ).cuda()  # TODO NOTE: category starts from 1 (master-chef-can=1, cheezeit-box=2)
        # import pdb;pdb.set_trace()
        # pred_r,pred_t,pred_c,emb,emb2 = estimator(img_masked,cloud,choose,index)
        pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

        # # test with clustering
        # emb1 = emb.cpu().numpy()
        # emb2=emb2.cpu().numpy()
        # from sklearn.cluster import KMeans
        # clus1 = KMeans(n_clusters=3).fit(emb1[0].transpose(1,0))
        # clus2 = KMeans(n_clusters=3).fit(emb2[0].transpose(1,0))
        # clus3 = KMeans(n_clusters=2).fit(emb2[0].transpose(1,0))
        # clus4 = KMeans(n_clusters=4).fit(emb2[0].transpose(1,0))
        # tmp1,tmp2,tmp3,tmp4= np.zeros((rmax-rmin)*(cmax-cmin)), np.zeros((rmax-rmin)*(cmax-cmin)), np.zeros((rmax-rmin)*(cmax-cmin)), np.zeros((rmax-rmin)*(cmax-cmin))
        # tmp1[choose.cpu().detach().numpy()]=clus1.labels_
        # tmp2[choose.cpu().detach().numpy()]=clus2.labels_
        # tmp3[choose.cpu().detach().numpy()]=clus3.labels_
        # tmp4[choose.cpu().detach().numpy()]=clus4.labels_
        # # import pdb;pdb.set_trace()

        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)
        points = cloud.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).detach().cpu().numpy()
        my_t = (points + pred_t)[which_max[0]].view(-1).detach().cpu().numpy()
        my_pred = np.append(my_r, my_t)
        pred_wo_refine[obj - 2] = my_pred.tolist()
        # import pdb;pdb.set_trace()
        for ite in range(0, 3):
            T = (
                torch.from_numpy(my_t.astype(np.float32))
                .cuda()
                .view(1, 3)
                .repeat(num_points, 1)
                .contiguous()
                .view(1, num_points, 3)
            )
            my_mat = quaternion_matrix(my_r)
            R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t

            new_cloud = torch.bmm((cloud - T), R).contiguous()
            pred_r, pred_t = refiner(new_cloud, emb, index)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).detach().cpu().numpy()
            my_t_2 = pred_t.view(-1).detach().cpu().numpy()
            my_mat_2 = quaternion_matrix(my_r_2)

            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array(
                [my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]]
            )

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

            pred_wo_refine[obj - 2] = my_pred.tolist()
        # import ipdb;ipdb.set_trace()
        # print(f'pred eye: obj {obj} | r {my_r} | t {my_t}')

    return pred_wo_refine


def get_mask(img, mask_estimator):
    segment = np.zeros(img.shape[:2])
    outputs = mask_estimator(
        img[:, :, ::-1]
    )  # NOTE: maskrcnn is trained with BGR images. img in param is RGB.
    masks = outputs["instances"].pred_masks.detach().cpu().numpy()
    classes = outputs["instances"].pred_classes.detach().cpu().numpy()
    del outputs
    for i, cls in enumerate(classes):
        segment[masks[i]] = cls
    return segment.astype(np.int32)


def load_vertices(ycb_id, obj_bank):
    vertices = read_obj(
        f"{YCB_BANK_DIR}/{obj_bank[ycb_id]}/textured.obj", decompose=False
    ).vertices
    return np.asarray(vertices)


import matplotlib.pyplot as plt

# global OBSIDX
# OBSIDX=-1
isii = -1
EYE_BASE_POSITION = (0.15, 0, 1.17)


class Observe_specific(Command):
    def __init__(self, entity, pose, vspace):
        self.entity = entity
        self.pose = pose
        self.vspace = vspace
        self.object_list = None

    def apply(self, state, **kwargs):
        base_pose = EYE_BASE_POSITION  # TODO adhoc
        pose = self.pose
        vspace = self.vspace
        target_pose = pose.value[0]

        (rgba, depth, segment), view_matrix, projection_matrix, _, _ = get_image(
            base_pose, target_pose, segment=True
        )

        debug = False  # True
        if debug:
            global isii

            isii = isii + 1
            plt.imsave(f"render/rgba.png", rgba)
            plt.imsave(f"render/rgba_{isii}.png", rgba)
            print(f"save image to {isii}")
        img_predpose = rgba[:, :, :3]
        """1) object detection & 6d pose estimation."""
        proj_matrix_real = np.asarray(projection_matrix).reshape(4, 4).transpose(1, 0)
        view_matrix_real = np.asarray(view_matrix).reshape(4, 4).transpose(1, 0)
        # print(f'GT mat  eye of obj {i}: {np.dot(np.linalg.inv(view_matrix_real),mat_world)}')

        depth_scaled = (depth - 0.5) * 2  # scale from 0-1 to -1~1
        xmap = np.array([[i for i in range(640)] for j in range(480)]) / 640.0 * 2 - 1.0
        ymap = (
            np.array([[480 - j for i in range(640)] for j in range(480)]) / 480.0 * 2
            - 1.0
        )
        normed_loc = np.concatenate(
            (
                xmap.reshape(1, -1),
                ymap.reshape(1, -1),
                depth_scaled.reshape(1, -1),
                np.ones((1, 640 * 480)),
            )
        )
        # loc in normalized space(-1~1)
        eye_loc = np.linalg.inv(proj_matrix_real).dot(normed_loc)
        world_loc = np.linalg.inv(view_matrix_real).dot(eye_loc)
        eye_loc = (
            (eye_loc / eye_loc[-1])[:3, :].reshape(3, 480, 640).transpose(1, 2, 0)
        )  # H x W x 3
        world_loc = (
            (world_loc / world_loc[-1])[:3, :].reshape(3, 480, 640).transpose(1, 2, 0)
        )

        segment = get_mask(rgba[:, :, :3], state.task.vis_handler["mask"])
        if debug:
            plt.imsave(f"render/seg.png", segment)
            plt.imsave(f"render/seg_{isii}.png", segment)

        obj2pose = get_pose6d(
            rgba,
            depth,
            eye_loc,
            segment,
            state.task.vis_handler["pose6d_init"],
            state.task.vis_handler["pose6d_refiner"],
        )

        entity_list = []
        for obj_i in obj2pose.keys():
            if (
                obj_i not in state.registered.keys()
                or state.registered[obj_i].pose_uncertain
            ):
                mat_eye = quaternion_matrix(obj2pose[obj_i][:4])
                mat_eye[0:3, 3] = obj2pose[obj_i][-3:]
                mat_eye[2, :] *= -1
                mat_eye[1, :] *= -1

                mat_world = np.linalg.inv(view_matrix_real).dot(mat_eye)
                tran_world = (mat_world[0, 3], mat_world[1, 3], mat_world[2, 3])
                mat_world[0:3, 3] = 0
                quat = quaternion_from_matrix(mat_world, True)
                quat_world = (quat[1], quat[2], quat[3], quat[0])  # WXYZ to XYZW
                if not vspace.point_in_space(tran_world):  # skip objects outside ROI
                    continue
                model = (
                    load_vertices(obj_i, state.task.obj_bank)
                    if obj_i not in state.registered.keys()
                    else state.registered[obj_i].model
                )  # N x 3
                # md = proj_matrix_real.dot(view_matrix_real.dot(np.concatenate((model,np.ones((1016,1))),1).transpose(1,0)))
                mask_proj = np.zeros((480, 640, 3))
                mask_given = (segment == obj_i + 2).astype(np.int32)
                # md = (md/md[-1])[:3,:].transpose(1,0) # M x 3
                img_length, img_width = 640, 480
                cam_fx = proj_matrix_real[0, 0] * img_length / 2.0
                cam_fy = proj_matrix_real[1, 1] * img_width / 2.0
                cam_cx = (proj_matrix_real[2, 0] - 1.0) * img_length / -2.0
                cam_cy = (proj_matrix_real[2, 1] + 1.0) * img_width / 2.0
                cam_mat = np.matrix(
                    [[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]
                )
                proj_r = mat_eye[0:3, 0:3]
                proj_r[1:, :] *= -1
                proj_t = np.asarray(obj2pose[obj_i][-3:])
                dist = np.zeros(5)
                imgpts, jac = cv2.projectPoints(model, proj_r, proj_t, cam_mat, dist)
                mask_proj = cv2.polylines(
                    mask_proj, np.int32([np.squeeze(imgpts)]), True, (255, 255, 255)
                )[:, :, 0].astype(np.int32)
                iou = (
                    float(np.logical_and(mask_proj, mask_given).sum())
                    / np.logical_or(mask_proj, mask_given).sum()
                )
                # import pdb;pdb.set_trace()
                if obj_i not in state.registered.keys():
                    # model = np.asarray(vertices_from_rigid(obj_i)) #TODO  adhoc
                    new_obj = load_pybullet(
                        f"{YCB_BANK_DIR}/{state.task.obj_bank[obj_i]}/textured.obj"
                    )  # TODO trick for grasping
                    set_pose(
                        new_obj,
                        (
                            tuple(np.asarray(tran_world) + np.asarray([0, 2, 0])),
                            quat_world,
                        ),
                    )  # TODO
                    obj_id_world = -1
                    least_distance = 10000
                    for center_world in state.task.pose2obj:
                        dist = (
                            (np.asarray(center_world) - np.asarray(tran_world)) ** 2
                        ).sum()
                        if dist < least_distance:
                            obj_id_world = state.task.pose2obj[center_world]
                            least_distance = dist
                    new_ent = Entity(
                        obj_id=obj_id_world,
                        obj_id_head=int(new_obj),
                        category=obj_i,
                        loc=(tran_world, quat_world),
                        model=model,
                        iou=iou,
                        obs_time=1,
                    )
                    add_text(
                        f"{new_ent}",
                        position=(0, 0, 0.1),
                        color=(0, 0, 0),
                        parent=new_ent.obj_id,
                    )  # position is the relative position between text and obj
                    add_text(
                        f"{new_ent}",
                        position=(0, 0, 0.1),
                        color=(0, 0, 0),
                        parent=new_obj,
                    )  # TODO trick
                    state.registered[obj_i] = new_ent
                else:
                    state.registered[obj_i].obs_time += 1
                    if state.registered[obj_i].obs_time >= OBS_TIME_THRESHOLD:
                        ps = world_loc[segment == obj_i + 2]
                        ps = ps - ps.mean(0)
                        std = np.std(ps)
                        mesh = create_mesh(
                            mesh_from_points(ps[(ps**2).sum(1) <= 0.5 * std])
                        )
                        set_point(
                            mesh, (1 + 0.5 * len(state.registered.keys()), 1, 1.5)
                        )
                        # pass # TODO generate mesh for pose
                    elif (
                        iou >= state.registered[obj_i].iou
                    ):  # TODO better metric than iou?
                        state.registered[obj_i].update_pose(
                            (tran_world, quat_world), fix_symbol=True
                        )
                        if (
                            iou >= OBS_IOU_THRESHOLD
                        ):  # state.registered[obj_i].loc == self.pose:
                            state.registered[obj_i].pose_uncertain = (
                                False  # TODO how to determine pose is accurate?
                            )
                entity_list.append(state.registered[obj_i])

        if debug:
            plt.imsave(f"render/predpose_{isii}.png", img_predpose)
        """
        2) data association
        """

        """
        3) update occupancy grid 
        """
        can_free_voxel, block_list = vspace.check_vis(
            camera_pos=base_pose,
            eye_coord_cloud=eye_loc,
            segment=segment,
            view_matrix_real=view_matrix_real,
            proj_matrix_real=proj_matrix_real,
            target_point=target_pose,
            entity_list=entity_list,
        )
        # print('=======Block list:',block_list,'==========')
        for block_obj in block_list.keys():
            if block_obj not in state.block_list.keys():
                state.block_list[block_obj] = block_list[block_obj]
            else:
                for block_value in block_list[block_obj]:
                    if block_value not in state.block_list[block_obj]:
                        state.block_list[block_obj].append(block_value)

        state.task.vspace.set_frees(can_free_voxel)
        self.object_list = obj2pose.keys()
        print(
            f"After command: len unobserved {len(self.vspace.occupied)}, \
                object list {self.object_list}"
        )
        # disconnect()
        # connect(
        vislist = []
        with LockRenderer():
            for voxel in vspace.occupied:
                vislist.append(
                    vspace.create_box(
                        (vspace.center_from_voxel(voxel), (0, 0, 0, 0.1)),
                        color=(1, 0, 0, 0.3),
                    )
                )
        wait_for_duration(3)
        with LockRenderer():
            for tmpbox in vislist:
                remove_body(tmpbox)
        wait_for_duration(1)
        yield

    def __repr__(self):
        return "{}({},{},{})".format(
            self.__class__.__name__,
            self.vspace,
            self.object_list,
            len(self.vspace.occupied),
        )


class Observe(Command):
    def __init__(self, pose, base_pose, vspace):
        self.vspace = vspace
        self.pose = pose
        self.base_pose = base_pose
        self.object_list = None

    def apply(self, state, **kwargs):
        base_pose = EYE_BASE_POSITION  # TODO adhoc
        pose = self.pose
        vspace = self.vspace
        target_pose = pose.value[0]

        (rgba, depth, segment), view_matrix, projection_matrix, _, _ = get_image(
            base_pose, target_pose, segment=True
        )

        debug = False  # True
        if debug:
            global isii

            isii = isii + 1
            # global OBSIDX
            # OBSIDX = isi
            plt.imsave(f"render/rgba.png", rgba)
            # plt.imsave(f'render/dep.png',depth)
            plt.imsave(f"render/rgba_{isii}.png", rgba)
            plt.imsave(f"render/dep_{isii}.png", depth)
            print(f"save image to {isii}")
            # import pdb
            # pdb.set_trace()
        img_predpose = rgba[:, :, :3]
        """1) object detection & 6d pose estimation."""
        proj_matrix_real = np.asarray(projection_matrix).reshape(4, 4).transpose(1, 0)
        view_matrix_real = np.asarray(view_matrix).reshape(4, 4).transpose(1, 0)
        # print(f'GT mat  eye of obj {i}: {np.dot(np.linalg.inv(view_matrix_real),mat_world)}')

        depth_scaled = (depth - 0.5) * 2  # scale from 0-1 to -1~1
        xmap = np.array([[i for i in range(640)] for j in range(480)]) / 640.0 * 2 - 1.0
        ymap = (
            np.array([[480 - j for i in range(640)] for j in range(480)]) / 480.0 * 2
            - 1.0
        )
        normed_loc = np.concatenate(
            (
                xmap.reshape(1, -1),
                ymap.reshape(1, -1),
                depth_scaled.reshape(1, -1),
                np.ones((1, 640 * 480)),
            )
        )
        # loc in normalized space(-1~1)
        eye_loc = np.linalg.inv(proj_matrix_real).dot(normed_loc)
        world_loc = np.linalg.inv(view_matrix_real).dot(eye_loc)

        eye_loc = (
            (eye_loc / eye_loc[-1])[:3, :].reshape(3, 480, 640).transpose(1, 2, 0)
        )  # H x W x 3
        world_loc = (
            (world_loc / world_loc[-1])[:3, :].reshape(3, 480, 640).transpose(1, 2, 0)
        )

        segment = get_mask(rgba[:, :, :3], state.task.vis_handler["mask"])
        if debug:
            plt.imsave(f"render/seg.png", segment)
            plt.imsave(f"render/seg_{isii}.png", segment)

        obj2pose = get_pose6d(
            rgba,
            depth,
            eye_loc,
            segment,
            state.task.vis_handler["pose6d_init"],
            state.task.vis_handler["pose6d_refiner"],
        )

        entity_list = []
        for obj_i in obj2pose.keys():
            if (
                obj_i not in state.registered.keys()
                or state.registered[obj_i].pose_uncertain
            ):
                mat_eye = quaternion_matrix(obj2pose[obj_i][:4])
                mat_eye[0:3, 3] = obj2pose[obj_i][-3:]
                mat_eye[2, :] *= -1
                mat_eye[1, :] *= -1

                mat_world = np.linalg.inv(view_matrix_real).dot(mat_eye)
                tran_world = (mat_world[0, 3], mat_world[1, 3], mat_world[2, 3])
                mat_world[0:3, 3] = 0
                quat = quaternion_from_matrix(mat_world, True)
                quat_world = (quat[1], quat[2], quat[3], quat[0])  # WXYZ to XYZW
                if not vspace.point_in_space(tran_world):  # skip objects outside ROI
                    continue
                model = (
                    load_vertices(obj_i, state.task.obj_bank)
                    if obj_i not in state.registered.keys()
                    else state.registered[obj_i].model
                )  # N x 3
                # md = proj_matrix_real.dot(view_matrix_real.dot(np.concatenate((model,np.ones((1016,1))),1).transpose(1,0)))
                mask_proj = np.zeros((480, 640, 3))
                mask_given = (segment == obj_i + 2).astype(np.int32)
                # md = (md/md[-1])[:3,:].transpose(1,0) # M x 3
                img_length, img_width = 640, 480
                cam_fx = proj_matrix_real[0, 0] * img_length / 2.0
                cam_fy = proj_matrix_real[1, 1] * img_width / 2.0
                cam_cx = (proj_matrix_real[2, 0] - 1.0) * img_length / -2.0
                cam_cy = (proj_matrix_real[2, 1] + 1.0) * img_width / 2.0
                cam_mat = np.matrix(
                    [[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]
                )
                proj_r = mat_eye[0:3, 0:3]
                proj_r[1:, :] *= -1
                proj_t = np.asarray(obj2pose[obj_i][-3:])
                dist = np.zeros(5)
                imgpts, jac = cv2.projectPoints(model, proj_r, proj_t, cam_mat, dist)
                mask_proj = cv2.polylines(
                    mask_proj, np.int32([np.squeeze(imgpts)]), True, (255, 255, 255)
                )[:, :, 0].astype(np.int32)
                iou = (
                    float(np.logical_and(mask_proj, mask_given).sum())
                    / np.logical_or(mask_proj, mask_given).sum()
                )
                # import pdb;pdb.set_trace()
                if obj_i not in state.registered.keys():
                    # model = np.asarray(vertices_from_rigid(obj_i)) #TODO  adhoc
                    new_obj = load_pybullet(
                        f"{YCB_BANK_DIR}/{state.task.obj_bank[obj_i]}/textured.obj"
                    )  # TODO trick for grasping
                    set_pose(
                        new_obj,
                        (
                            tuple(np.asarray(tran_world) + np.asarray([0, 2, 0])),
                            quat_world,
                        ),
                    )  # TODO
                    obj_id_world = -1
                    least_distance = 10000
                    for center_world in state.task.pose2obj:
                        dist = (
                            (np.asarray(center_world) - np.asarray(tran_world)) ** 2
                        ).sum()
                        if dist < least_distance:
                            obj_id_world = state.task.pose2obj[center_world]
                            least_distance = dist
                    new_ent = Entity(
                        obj_id=obj_id_world,
                        obj_id_head=int(new_obj),
                        category=obj_i,
                        loc=(tran_world, quat_world),
                        model=model,
                        iou=iou,
                        obs_time=1,
                    )
                    add_text(
                        f"{new_ent}",
                        position=(0, 0, 0.1),
                        color=(0, 0, 0),
                        parent=new_ent.obj_id,
                    )  # position is the relative position between text and obj
                    add_text(
                        f"{new_ent}",
                        position=(0, 0, 0.1),
                        color=(0, 0, 0),
                        parent=new_obj,
                    )  # TODO trick
                    state.registered[obj_i] = new_ent
                else:
                    state.registered[obj_i].obs_time += 1
                    if state.registered[obj_i].obs_time >= OBS_TIME_THRESHOLD:
                        ps = world_loc[segment == obj_i + 2]
                        ps = ps - ps.mean(0)
                        std = np.std(ps)
                        mesh = create_mesh(
                            mesh_from_points(ps[(ps**2).sum(1) <= 0.5 * std])
                        )
                        set_point(
                            mesh, (1 + 0.5 * len(state.registered.keys()), 1, 1.5)
                        )
                        # pass # TODO generate mesh for pose
                    elif (
                        iou >= state.registered[obj_i].iou
                    ):  # TODO better metric than iou?
                        state.registered[obj_i].update_pose(
                            (tran_world, quat_world), fix_symbol=True
                        )
                        if (
                            iou >= OBS_IOU_THRESHOLD
                        ):  # state.registered[obj_i].loc == self.pose:
                            state.registered[obj_i].pose_uncertain = (
                                False  # TODO how to determine pose is accurate?
                            )
                entity_list.append(state.registered[obj_i])
        if debug:
            plt.imsave(f"render/predpose_{isii}.png", img_predpose)
        """
        2) data association
        """

        """
        3) update occupancy grid 
        """
        can_free_voxel, block_list = vspace.check_vis(
            camera_pos=base_pose,
            eye_coord_cloud=eye_loc,
            segment=segment,
            view_matrix_real=view_matrix_real,
            proj_matrix_real=proj_matrix_real,
            target_point=target_pose,
            entity_list=entity_list,
        )
        print("Block list:", block_list)
        for block_obj in block_list.keys():
            if block_obj not in state.block_list.keys():
                state.block_list[block_obj] = block_list[block_obj]
            else:
                for block_value in block_list[block_obj]:
                    if block_value not in state.block_list[block_obj]:
                        state.block_list[block_obj].append(block_value)

        state.task.vspace.set_frees(can_free_voxel)
        self.object_list = obj2pose.keys()
        print(
            f"After command: len unobserved {len(self.vspace.occupied)}, \
                object list {self.object_list}"
        )
        # disconnect()
        # connect(
        vislist = []
        with LockRenderer():
            for voxel in vspace.occupied:
                vislist.append(
                    vspace.create_box(
                        (vspace.center_from_voxel(voxel), (0, 0, 0, 0.1)),
                        color=(1, 0, 0, 0.3),
                    )
                )
        wait_for_duration(3)
        with LockRenderer():
            for tmpbox in vislist:
                remove_body(tmpbox)
        wait_for_duration(1)
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
