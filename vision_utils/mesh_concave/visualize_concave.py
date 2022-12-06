import argparse
import os
import os.path as osp
import sys

import numpy as np
import pybullet as p

sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
        #'pddlstream/examples/pybullet/utils',
        #'../ltamp_pr2',
    ]
)

from collections import Counter

from environments import Z_EPSILON, create_default_env, create_floor_object
from pybullet_tools.utils import (
    PI,
    TEMP_DIR,
    CameraImage,
    Euler,
    LockRenderer,
    Point,
    Pose,
    add_data_path,
    add_text,
    connect,
    draw_point,
    enable_gravity,
    enable_real_time,
    get_point,
    load_pybullet,
    set_all_static,
    set_camera_pose,
    set_point,
    set_pose,
    stable_z,
    wait_for_user,
    wait_if_gui,
)

connect(use_gui=True)
add_data_path()
floor = create_floor_object()
set_camera_pose(camera_point=[1.0, -1.0, 1.5])
enable_gravity()
enable_real_time()

ycb_dir = "vision_utils/ycb_models"
meshes_dir = "vision_utils/mesh_concave"
folders = os.listdir(meshes_dir)
test_obj_list = ["040_large_marker", "051_large_clamp", "037_scissors"]

# meshes_folders = [osp.join(meshes_dir,folder) for folder in folders]
meshes_folders = [osp.join(meshes_dir, "precomputed")]
x, y = 0, 0
max_y = 1.5
step = 0.4

meshes = []
for mesh_folder in meshes_folders:
    if (
        osp.basename(mesh_folder) == "6" or osp.basename(mesh_folder) == "5"
    ):  # will lead to segmentation fault
        print(
            f"skip {mesh_folder} to avoid segmentation fault in pybullet, try visualize using meshlab"
        )
        continue
    if not osp.isdir(mesh_folder):
        continue
    mesh_files = os.listdir(mesh_folder)
    # bowls = [mesh_file for mesh_file in mesh_files if 'bowl' in mesh_file]
    bowls = mesh_files
    for bowl_name in bowls:
        if bowl_name[-3:] == "obj":
            path = osp.join(mesh_folder, bowl_name)
            if "vhacd" not in path and bowl_name[:-4] + "_vhacd.obj" not in bowls:
                bowl_name_out = bowl_name[:-4] + "_vhacd.obj"
                p.vhacd(
                    osp.join(mesh_folder, bowl_name),
                    osp.join(mesh_folder, bowl_name_out),
                    osp.join(TEMP_DIR, "vhacd_tmp_log"),
                    alpha=0.04,
                    resolution=50000,
                )
                bowl_name = bowl_name_out
                path = osp.join(mesh_folder, bowl_name)
            elif "vhacd" not in path and bowl_name[:-4] + "_vhacd.obj" in bowls:
                continue
            print(path)

            try:
                bowl = load_pybullet(path, mass=1)
                bloc = (x, y, 0.2)
                meshes.append(bowl)
                set_point(bowl, bloc)
                add_text("/".join(path.split("/")[2:]), parent=bowl)
                y += step
                if y > max_y:
                    y = 0
                    x += step
            except:
                print(f"load {path} failed. skipped")


def drop_loc(base_loc):
    return tuple(np.add((base_loc), (0, 0, 0.2)))


marker_path = osp.join(ycb_dir, test_obj_list[0], "textured.obj")
markers = []
for mesh in meshes:
    pen = load_pybullet(marker_path, mass=0.2)
    base_loc = drop_loc(get_point(mesh))
    set_pose(pen, (base_loc, (1, 1, 1, 1)))
    markers.append(pen)

while True:
    key = wait_for_user()
    print("press 'c' to continue")
    if key == "c":
        for i, mesh in enumerate(meshes):
            set_pose(markers[i], (drop_loc(get_point(mesh)), (1, 1, 1, 1)))
    else:
        break
