#!/usr/bin/env python3

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pybullet as p

# NOTE(caelan): must come before other imports
sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
        #'pddlstream/examples/pybullet/utils',
        #'../ltamp_pr2',
    ]
)

import itertools
import json
from collections import Counter

import cv2
import trimesh
from open_world.estimation.bounding import estimate_oobb
from open_world.estimation.dnn import init_sc, init_seg
from open_world.estimation.geometry import (estimate_surface_mesh,
                                            filter_visible,
                                            project_base_points, refine_shape)
from open_world.simulation.environment import create_ycb
from open_world.simulation.lis import (CAMERA_MATRIX, YCB_COLORS, YCB_MASSES,
                                       Z_EPSILON, get_ycb_obj_path)
from pybullet_tools.pr2_problems import create_table
from pybullet_tools.utils import (CLIENT, INF, PI, STATIC_MASS, TAN, TEMP_DIR,
                                  WHITE, CameraImage, Euler, Mesh, Point, Pose,
                                  aabb_from_points, aabb_union, connect,
                                  create_box, create_collision_shape,
                                  create_obj, create_plane,
                                  create_visual_shape,
                                  dimensions_from_camera_matrix,
                                  disable_gravity, disconnect, enable_gravity,
                                  get_field_of_view, get_image,
                                  get_image_at_pose, get_mesh_data, get_pose,
                                  get_visual_data, invert, matrix_from_quat,
                                  mesh_from_points, multiply, remove_body,
                                  set_all_static, set_euler, set_point,
                                  set_pose, stable_z, tform_points)


def cloud_from_depth(camera_matrix, depth, max_depth=10.0):
    width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
    xmap = np.array(
        [[i for i in range(width)] for _ in range(height)]
    )  # 0 ~ width. hxw
    ymap = np.array(
        [[height - j for _ in range(width)] for j in range(height)]
    )  # 0 ~ height. hxw
    homogeneous_coord = np.concatenate(
        [xmap.reshape(1, -1), ymap.reshape(1, -1), np.ones((1, height * width))]
    )  # 3 x (hw)
    rays = np.linalg.inv(camera_matrix).dot(homogeneous_coord)
    point_cloud = depth.reshape(1, height * width) * rays
    point_cloud = point_cloud.transpose(1, 0).reshape(height, width, 3)
    point_cloud[point_cloud[:, :, 2] > max_depth] = 0  # clamp large values
    return point_cloud


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--viewer", action="store_true", help="")
    parser.add_argument(
        "-scm",
        "--shape_completion_model",
        type=str,
        default="msn",
        choices=["msn", "atlas"],
        help="select model for shape completion. msn | atlas",
    )
    # http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
    parser.add_argument(
        "-a",
        "--alpha_shape",
        action="store_true",
        help="create alpha shape to handle concave meshes(should be used together with vhacd)",
    )
    parser.add_argument(
        "-seg",
        "--segmentation",
        action="store_true",
        help="set to True to use DNN for segmentation",
    )
    parser.add_argument(
        "-segm",
        "--segmentation_model",
        type=str,
        default="all",
        choices=["maskrcnn", "uois", "all"],
        help="select model for segmentation. maskrcnn | uois",
    )
    # parser.add_argument('-p', '--pose', action='store_true',
    # help='use dnn for 6d pose estimation') # don't need 6d pose currently
    return parser


def init_networks(args):
    sc_network = seg_network = None
    sc_network = init_sc(branch=args.shape_completion_model)
    if args.segmentation:
        seg_network = init_seg(
            branch=args.segmentation_model,
            maskrcnn_rgbd=args.maskrcnn_rgbd,
            post_classifier=args.fasterrcnn_detection,
        )
    return seg_network, sc_network


################################################################################


def get_aabb_from_cloud(cloud_a, cloud_b):
    aabb_a = aabb_from_points(cloud_a)  # camera frame
    aabb_b = aabb_from_points(cloud_b)  # camera_frame
    aabb_final = aabb_union([aabb_a, aabb_b])
    aabb_lower, aabb_upper = aabb_final
    # bounds = np.concatenate((aabb_lower.reshape(1,3),aabb_upper.reshape(1,3)),axis=0)
    bounds = np.concatenate((aabb_lower, aabb_upper), axis=0)  # (6,)
    return bounds


def trimeshhull_from_cloud(cloud):
    mesh_tmp = mesh_from_points(cloud)
    return trimesh.Trimesh(vertices=mesh_tmp.vertices, faces=mesh_tmp.faces).convex_hull


def get_iou_fromvox(vox1, vox2):
    vox1_densemat = vox1.matrix
    vox2_densemat = vox2.matrix
    assert vox1_densemat.shape == vox2_densemat.shape

    intersect = vox1_densemat & vox2_densemat
    union = vox1_densemat | vox2_densemat
    voxel_iou = intersect.sum() / union.sum()
    # print(f'iou is {voxel_iou}')
    return float(voxel_iou)


def vox_from_trimeshhull(trimesh_hull, bounds):
    # https://github.com/mikedh/trimesh/blob/77d10ffda2/examples/voxel.py

    # vanilla pip installation does not come with binvox. Download and add executable file to path
    # https://github.com/mikedh/trimesh/blob/master/trimesh/exchange/binvox.py

    # exact: any voxel with part of a triangle gets set. Does not use
    #     graphics card.
    # max grid size is 1024 if not using exact

    # fill - don't use 'base'
    return trimesh_hull.voxelized(
        method="binvox", exact=True, dimension=128, bounding_box=bounds
    ).fill(method="holes")


def get_iou_from_cldhull(cld1, cld2):
    mesh1 = (
        trimeshhull_from_cloud(cld1)
        if not isinstance(cld1, trimesh.base.Trimesh)
        else cld1
    )
    mesh2 = (
        trimeshhull_from_cloud(cld2)
        if not isinstance(cld2, trimesh.base.Trimesh)
        else cld2
    )
    cld1 = mesh1.vertices
    cld2 = mesh2.vertices
    bounds = get_aabb_from_cloud(cld1, cld2)
    vox1 = vox_from_trimeshhull(mesh1, bounds)
    vox2 = vox_from_trimeshhull(mesh2, bounds)
    return get_iou_fromvox(vox1, vox2)


def get_mesh_geometry_scaled(path, scale=np.ones(3)):
    return {
        "shapeType": p.GEOM_MESH,
        "fileName": path,
        "meshScale": scale,
    }


def create_scaled_ycb(
    name, use_concave=False, mass_scale=1.0, scale=np.ones(1), path=None
):
    if path is None:
        concave_ycb_path = get_ycb_obj_path(name, use_concave=use_concave)
        ycb_path = get_ycb_obj_path(name)
    else:
        concave_ycb_path = ycb_path = path
    calculated_color = WHITE

    full_name = os.path.basename(os.path.dirname(ycb_path))
    mass = mass_scale * YCB_MASSES[name] if name in YCB_MASSES.keys() else 1
    mesh = trimesh.load(ycb_path)
    visual_geometry = get_mesh_geometry_scaled(
        ycb_path, scale=scale
    )  # TODO: randomly transform
    collision_geometry = get_mesh_geometry_scaled(concave_ycb_path, scale=scale)
    geometry_pose = Pose(point=-mesh.center_mass)
    collision_id = create_collision_shape(collision_geometry, pose=geometry_pose)
    visual_id = create_visual_shape(
        visual_geometry, color=calculated_color, pose=geometry_pose
    )
    # collision_id, visual_id = create_shape(geometry, collision=True, color=WHITE)
    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        # basePosition=[0., 0., 0.1],
        physicsClientId=CLIENT,
    )
    return body


def get_random(scale=1.0, offset=0.0):
    return np.random.rand(1)[0] * scale + offset


def visualize_and_save(clds, prefixes, idx):
    assert len(clds) == len(prefixes)
    for i, cld in enumerate(clds):
        trimeshhull_from_cloud(cld).export(f"{prefixes[i]}_{idx}.obj")


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_image_at_pose_target(
    camera_loc, camera_matrix, target_point, far=5.0, **kwargs
):
    width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
    _, vertical_fov = get_field_of_view(camera_matrix)
    camera_point = camera_loc
    return get_image(
        camera_point,
        target_point,
        width=width,
        height=height,
        vertical_fov=vertical_fov,
        far=far,
        **kwargs,
    )


def print_to_file(f, *args):
    print(*args)
    print(*args, file=f)


def get_valid_image(table_h):
    point_cam = (
        get_random(-0.1, -0.3),
        get_random(0.5, -0.25),
        get_random(1, table_h + 0.02),
    )  # random height.
    target_point = (0, 0, table_h + 1e-3)
    camera_image = get_image_at_pose_target(
        point_cam,
        CAMERA_MATRIX,
        target_point,
        tiny=False,
        segment=True,
        segment_links=False,
    )
    return camera_image


def pose_vec(objects):
    return list(itertools.chain(*[get_pose(obj)[0] for obj in objects]))


def simulate_until_stable(objects, epsilon=1e-4, steps=10, maxiter=40):
    # copy from run_planner.py (w/o robot)
    obj_poses = pose_vec(objects)
    enable_gravity()
    for _ in range(steps):
        p.stepSimulation()
    trial_iter = 0
    while (
        np.linalg.norm(np.array(obj_poses) - np.array(pose_vec(objects))) > epsilon
    ) and (trial_iter < maxiter):
        obj_poses = pose_vec(objects)
        for _ in range(steps):
            p.stepSimulation()
        trial_iter += 1
    disable_gravity()


def create_clutter(table_h):
    # # 1
    # # unseen_categories = [1,12,13,14,15,17,18,26]
    # categories = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,21,26,29,36,48] # scaled to make it different
    # basedir = './models/srl/ycb/'
    # test_dirs = os.listdir(basedir)
    # # objlist = [os.path.join(basedir,x,'textured.obj') for x in test_dirs if '_' in x]
    # objlist = [os.path.join(basedir,x,'textured.obj') for x in test_dirs if '_' in x and int(x.split('_')[0]) in categories]
    # scale_fac = 1.

    # 2
    shapenet_categories = ["02876657", "02946921", "04401088"]  # bottle, can, phone
    basedir = "../dsr/object_models/shapenet"
    test_dirs = os.walk(basedir)
    objlist = [
        os.path.join(x, "model_com.obj")
        for (x, _, filenames) in test_dirs
        if len(x.split("shapenet")[1]) > 30
        and x.split("shapenet")[1].split("/")[1] in shapenet_categories
    ]
    scale_fac = 0.2

    # 3
    graspnet_categories = [
        10,
        11,
        12,
        13,
        14,
        16,
        17,
    ]  # correspond to unseen ycb ids [1,12,13,14,15,17,18] (original ycb meshes have holes so using graspnet models instead)
    # graspnet_categories = [2,4,16] # template shapes(box/can,cylinder,sphere) (original ycb meshes have holes so using graspnet models instead)
    basedir = "../graspnet/models_new/models/"
    test_dirs = os.listdir(basedir)
    objlist.extend(
        [
            os.path.join(basedir, x, "nontextured_simplified_2.obj")
            for x in test_dirs
            if len(x) == 3 and int(x) in graspnet_categories
        ]
    )
    # objlist=[os.path.join(basedir,x,'nontextured_simplified_2.obj') for x in test_dirs if len(x)==3 and int(x) in graspnet_categories]

    # print(f'objlist is {objlist}')
    clutter_number = np.random.randint(5) + 5  # 1
    selected_objects = []
    for i in range(clutter_number):
        obj_type = np.random.choice(objlist)
        scale_fac = 0.2 if "shapenet" in obj_type else 1.0
        rand_scalex, rand_scaley, rand_scalez = (
            scale_fac,
            scale_fac,
            scale_fac,
        )  # get_random(scale_fac,.5*scale_fac), get_random(scale_fac,.5*scale_fac), get_random(scale_fac,.5*scale_fac)
        body = create_scaled_ycb(
            None,
            mass_scale=1.0,
            scale=(rand_scalex, rand_scaley, rand_scalez),
            path=obj_type,
        )
        selected_objects.append(body)

        rand_locx, rand_locy = get_random(0.5, -0.25), get_random(1, -0.5)
        rand_euler = np.random.rand(3)
        set_point(body, (rand_locx, rand_locy, table_h + 0.1))
        set_euler(body, rand_euler)
        simulate_until_stable(selected_objects)
        # print(obj_type)
    set_all_static()
    return (
        selected_objects,
        body,
        [obj_type, rand_scalex, rand_scaley, rand_scalez],
    )  # return last object(not burried)


def get_trimesh_from_transform(test_body, obj_path, sx, sy, sz):
    so3 = matrix_from_quat(get_pose(test_body)[1])
    se3 = np.eye(4)
    se3[:3, :3] = so3
    se3[:3, 3] = get_pose(test_body)[0]

    gt_mesh = trimesh.load(obj_path)
    scale_transform = np.eye(4)
    scale_transform[:3, :3] *= np.asarray([sx, sy, sz])
    center_mass = gt_mesh.center_mass
    scale_transform[:3, 3] = -center_mass
    gt_mesh.apply_transform(scale_transform)
    gt_mesh.apply_transform(se3)
    return gt_mesh


def main(optimize=False):
    np.set_printoptions(
        precision=3, threshold=3, suppress=True
    )  # , edgeitems=1) #, linewidth=1000)
    parser = create_parser()
    args = parser.parse_args()

    seg_network, sc_network = init_networks(args)

    test_num = 2000
    savedir = "eval_numobj5-10_unseenshapes_smallgrid"
    cluttered = True
    # shape_template = 'tomato_soup_can'

    ensure_dir(savedir)
    ensure_dir(f"{savedir}/meshes")
    os.system(f"cp {__file__} {savedir}")
    logfile = open(f"{savedir}/log", "w+")

    stat = []
    scene_i = 0
    connect(use_gui=args.viewer)
    floor = create_plane(mass=STATIC_MASS, color=TAN)
    while True:
        rand_th = get_random(0.5, 0.5)  # 0.5~1
        table = create_table(height=rand_th)  # top_color=LIGHT_GREY,

        if not cluttered:
            raise DeprecationWarning
            # structured. only 1 object. simple geometry
            rand_w, rand_l, rand_h = (
                get_random(1, 0.5),
                get_random(1, 0.5),
                get_random(0.5, 0.5),
            )
            test_body = create_scaled_ycb(
                shape_template, mass_scale=1.0, scale=(rand_w, rand_l, rand_h)
            )
            rand_locx, rand_locy, rand_locz = (
                get_random(0.5, -0.25),
                get_random(1, -0.5),
                stable_z(test_body, table),
            )
            rand_euler = get_random() * np.pi
            set_pose(
                test_body,
                Pose(
                    Point(x=rand_locx, y=rand_locy, z=rand_locz), Euler(yaw=rand_euler)
                ),
            )
        else:
            (
                all_objects,
                test_body,
                [test_body_objpath, test_body_sx, test_body_sy, test_body_sz],
            ) = create_clutter(rand_th)

        camera_image = get_valid_image(rand_th)
        obj_mask = camera_image.segmentationMaskBuffer[..., 0] == test_body
        if (
            obj_mask[0].any()
            | obj_mask[-1].any()
            | obj_mask[:, 0].any()
            | obj_mask[:, -1].any()
            | (obj_mask.sum() == 0)
        ):
            # object not fully in the fov
            disconnect()
            connect(use_gui=args.viewer)  # TODO pybullet memory error
            floor = create_plane(mass=STATIC_MASS, color=TAN)
            continue
            # camera_image = get_valid_image(rand_th)
            # obj_mask = camera_image.segmentationMaskBuffer[...,0]==test_body

        # gtmesh_from_cld = trimeshhull_from_cloud(tform_points(multiply(get_pose(test_body)),get_mesh_data(test_body)[1]))
        # trimesh.Scene([gt_mesh, gtmesh_from_cld]).show()

        cv2.imwrite(f"temp_meshes/sctest.png", camera_image[0][..., :3][..., ::-1])

        cloud_raw = cloud_from_depth(
            camera_image.camera_matrix, camera_image.depthPixels
        )[obj_mask]
        cloud_raw[..., 1] *= -1  # pybullet coordinate, camera
        cloud_raw = np.asarray(
            tform_points(multiply(camera_image.camera_pose), cloud_raw)
        )  # world frame

        record = {}
        record["table_wlh"] = (
            0.6,
            1.2,
            rand_th,
        )  # .6,1.2 are the default wl # TODO don't use constant
        if not cluttered:
            record["obj_geom"] = shape_template
            record["obj_scale"] = (rand_w, rand_l, rand_h)
        # record['obj_loc'] = (rand_locx, rand_locy, rand_locz, rand_euler) # cloud_gt
        record["camera_pose"] = camera_image.camera_pose
        record["camera_matrix"] = camera_image.camera_matrix.tolist()

        # ensure_dir(f'{savedir}/{scene_i}')
        cv2.imwrite(f"{savedir}/obs_{scene_i}.png", camera_image[0][..., :3][..., ::-1])
        cv2.imwrite(f"{savedir}/seg_{scene_i}.png", obj_mask.astype(np.uint8) * 255)

        # cloud_gt = tform_points(multiply(invert(camera_image.camera_pose), get_pose(test_body)),get_mesh_data(test_body)[1]) # camera
        cloud_gt = tform_points(
            multiply(get_pose(test_body)), get_mesh_data(test_body)[1]
        )  # world
        if cluttered:
            trimesh_gt = get_trimesh_from_transform(
                test_body, test_body_objpath, test_body_sx, test_body_sy, test_body_sz
            )
            if get_iou_from_cldhull(cloud_gt, trimesh_gt) < 0.85:
                trimesh_gt = trimesh_gt.convex_hull
        # import pdb;pdb.set_trace()
        # trimesh.Scene([voxelized_rawconvexhull_trimesh,voxelized_gtconvexhull_trimesh, mesh_rawconvexhull_trimesh, mesh_gtconvexhull_trimesh]).show()

        try:
            # if True:
            min_z = np.min(cloud_raw, axis=0)[2]
            surface_pose = Pose(Point(z=min_z))  # TODO: apply world_frame here
            labeled_cluster = tform_points(invert(surface_pose), cloud_raw)
            points = labeled_cluster.copy()
            # points = cloud_raw.copy()
            obj_oobb = estimate_oobb(points)  # , min_z=min_z)
            origin_pose = obj_oobb.pose  # TODO: adjust pose to be the base
            points_origin = tform_points(invert(origin_pose), points)
            max_z = INF

            base_origin = tform_points(
                invert(origin_pose), project_base_points(points, min_z=0, max_z=max_z)
            )
            vertices_schull_untransformed = refine_shape(
                sc_network, points_origin, use_points=True
            ).vertices

            cloud_proj = tform_points(multiply(surface_pose, origin_pose), base_origin)
            iou_proj2gt = (
                get_iou_from_cldhull(cloud_proj, trimesh_gt)
                if cluttered
                else get_iou_from_cldhull(cloud_proj, cloud_gt)
            )

            cloud_sc = tform_points(
                multiply(surface_pose, origin_pose), vertices_schull_untransformed
            )
            iou_sc2gt = (
                get_iou_from_cldhull(cloud_sc, trimesh_gt)
                if cluttered
                else get_iou_from_cldhull(cloud_sc, cloud_gt)
            )

            cloud_scfil_camframe = filter_visible(
                vertices_schull_untransformed,
                multiply(surface_pose, origin_pose),
                camera_image,
            )
            cloud_scfil = tform_points(
                multiply(surface_pose, origin_pose), cloud_scfil_camframe
            )
            iou_scfil2gt = (
                get_iou_from_cldhull(cloud_scfil, trimesh_gt)
                if cluttered
                else get_iou_from_cldhull(cloud_scfil, cloud_gt)
            )

            cloud_scfilproj = tform_points(
                multiply(surface_pose, origin_pose),
                np.vstack([cloud_scfil_camframe, base_origin]),
            )
            # cloud_scfilproj = np.vstack([cloud_scfil, cloud_proj])
            iou_scfilproj2gt = (
                get_iou_from_cldhull(cloud_scfilproj, trimesh_gt)
                if cluttered
                else get_iou_from_cldhull(cloud_scfilproj, cloud_gt)
            )

            iou_raw2gt = (
                get_iou_from_cldhull(cloud_raw, trimesh_gt)
                if cluttered
                else get_iou_from_cldhull(cloud_raw, cloud_gt)
            )
        except Exception as e:
            print("-" * 20)
            print(f"ERROR in computing iou. recompute scene {scene_i}")
            print(e)
            print("-" * 20)
            disconnect()
            connect(use_gui=args.viewer)  # TODO pybullet memory error
            floor = create_plane(mass=STATIC_MASS, color=TAN)
            continue

        record["cloud_gt"] = cloud_gt
        record["cloud_raw"] = cloud_raw.tolist()
        record["iou_raw2gt"] = iou_raw2gt
        record["cloud_proj"] = cloud_proj
        record["iou_proj2gt"] = iou_proj2gt
        record["cloud_sc"] = cloud_sc
        record["iou_sc2gt"] = iou_sc2gt
        record["cloud_scfil"] = cloud_scfil
        record["iou_scfil2gt"] = iou_scfil2gt
        record["cloud_scfilproj"] = cloud_scfilproj
        record["iou_scfilproj2gt"] = iou_scfilproj2gt

        print_to_file(logfile, f"scene {scene_i}")
        print_to_file(
            logfile, iou_raw2gt, iou_proj2gt, iou_sc2gt, iou_scfil2gt, iou_scfilproj2gt
        )
        stat.append(record)

        remove_body(table)
        remove_body(test_body)

        if scene_i % 10 == 0:
            cldtypes = ["gt", "raw", "proj", "sc", "scfiltered", "scfilteredproj"]
            visualize_and_save(
                [
                    cloud_gt,
                    cloud_raw,
                    cloud_proj,
                    cloud_sc,
                    cloud_scfil,
                    cloud_scfilproj,
                ],
                [f"{savedir}/meshes/mesh_{cldtype}" for cldtype in cldtypes],
                scene_i,
            )

        scene_i += 1

        disconnect()
        connect(use_gui=args.viewer)  # TODO pybullet memory error
        floor = create_plane(mass=STATIC_MASS, color=TAN)

        if scene_i == test_num:
            print_to_file(logfile, f"Test finished. {test_num} samples.")
            print_to_file(
                logfile,
                f"vIOU raw - gt:\t {np.mean([rec['iou_raw2gt'] for rec in stat])}",
            )
            print_to_file(
                logfile,
                f"vIOU projbase - gt\t: {np.mean([rec['iou_proj2gt'] for rec in stat])}",
            )
            print_to_file(
                logfile,
                f"vIOU sc - gt\t: {np.mean([rec['iou_sc2gt'] for rec in stat])}",
            )
            print_to_file(
                logfile,
                f"vIOU scfil - gt\t: {np.mean([rec['iou_scfil2gt'] for rec in stat])}",
            )
            print_to_file(
                logfile,
                f"vIOU scfilproj - gt\t: {np.mean([rec['iou_scfilproj2gt'] for rec in stat])}",
            )
            break

    with open(f"{savedir}/stat.json", "w") as fout:
        json.dump(stat, fout)
    logfile.close()
    disconnect()


if __name__ == "__main__":
    main()
