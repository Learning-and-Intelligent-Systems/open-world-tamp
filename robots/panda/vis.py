import copy
import pickle

import numpy as np
import open3d as o3d
from PIL import Image
from run_estimator import cloud_from_depth

# import rospy
# import tf2_ros
# from tf2_geometry_msgs import PoseStamped
#
# tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
# tf_listener = tf2_ros.TransformListener(tf_buffer)

# export PYTHONPATH=$PYTHONPATH:/home/honda/catkin_ws/src/owt:/home/honda/catkin_ws/src/owt/pybullet-planning
camera_lookup = {"032622074588": "A", "028522072401": "B", "032622073024": "C"}
camera_poses = {
    "032622074588": ((0, -105, 0), (0.836548, 0.396939, -0.139762, -0.350845)),
    "028522072401": ((0, 0, 0), (0.433279, 0.815291, -0.333735, -0.190239)),
    # '032622073024':(, quat_from_euler())
}
sns = list(camera_lookup.keys())


# TODO Figure out how the scene is set up.
# Get Extrinsic camera properties from that
# Figure out how to make depth real instead of *3
# Get third camera from extrinsics using configuration


def generate_pointcloud(data, depth_scale=1):

    # width, height = map(int, dimensions_from_camera_matrix(data['intrinsics'][0]))
    width, height = data["depth"].shape[1], data["depth"].shape[0]
    new_depth = (data["depth"] * 255.0) / np.max(data["depth"])
    dim = Image.fromarray(new_depth)
    dim = dim.resize((width, height))
    depth_data = np.asarray(data["depth"]) / 3000

    rim = Image.fromarray(data["rgb"])
    rim = rim.resize((width, height))
    rgb_data = np.asarray(rim)

    point_cloud = cloud_from_depth(
        data["intrinsics"][0], depth_data, max_depth=float("inf")
    )
    return point_cloud, rgb_data, depth_data


if __name__ == "__main__":
    pcds = []
    scene_name = "scene_1"
    with open("./panda/panda_dataset/" + scene_name + ".pkl", "rb") as handle:
        datas = pickle.load(handle)

    for data in datas:
        pcd = generate_pointcloud(data)
        pcds.append(pcd)

    def plot_registered():
        current_transformation = np.identity(4)

        pcds[0].estimate_normals()
        pcds[1].estimate_normals()

        # point cloud registration
        result_icp = o3d.pipelines.registration.registration_icp(
            pcds[0],
            pcds[1],
            20,
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        newpc0 = copy.deepcopy(pcds[0])
        newpc0.transform(result_icp.transformation)

        o3d.visualization.draw_geometries([newpc0, pcds[1]])
