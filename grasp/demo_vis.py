import os
import pickle
import time

import mayavi.mlab as mlab
import open3d as o3d
from dotenv import load_dotenv
from minio import Minio

from grasp.graspnet.utils.visualization_utils import *

load_dotenv()
minioClient = Minio(
    "ceph.csail.mit.edu",
    access_key=os.environ["S3ACCESS"],
    secret_key=os.environ["S3SECRET"],
    secure=True,
)

# # Read the data from minio and display it in mayavi


def read_from_minio(file):
    tmp_folder = "./tmp_minio/"
    if not os.path.isdir(tmp_folder):
        os.mkdir(tmp_folder)

    tmp_path = tmp_folder + file

    # s to get objects in folder
    minioClient.fget_object("aidan_bucket", file, tmp_path)

    # Write data to a pickle file
    with open(tmp_path, "rb") as handle:
        b = pickle.load(handle)
    return b


data = read_from_minio("003_cracker_box.pkl")
mlab.figure(bgcolor=(1, 1, 1))


total = np.concatenate([data["pc"], data["pc_colors"]], axis=1)
draw_scene(
    data["pc"],
    pc_color=data["pc_colors"],
    grasps=data["grasps"],
    grasp_scores=data["grasp_scores"],
    show_gripper_mesh=False,
)


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(data["pc"])

# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
# print(ind)
# inliers = pcd.select_by_index(ind)
# o3d.visualization.draw_geometries([inliers])


print("close the window to continue to next object . . .")
mlab.show()
