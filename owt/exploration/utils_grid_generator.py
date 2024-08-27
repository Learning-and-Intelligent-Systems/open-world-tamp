import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

ratio = 40
world_length = 40

dimensions_world = [world_length, world_length]
dimensions_pixel = [int(world_length * ratio), int(world_length * ratio)]

min_bounds = [-world_length / 2, -world_length / 2]
max_bounds = [world_length / 2, world_length / 2]

lower = np.array([-0.4083206145763397, -0.3692174541962325, -0.0030000009417533724])
upper = np.array([0.6765739326727533, 0.34854376903233264, 1.5969516277868887])


# Change 0.5 to 0.8
dimensions_movo = (abs(lower) + abs(upper)) * 0.6

MOVO_LENGTH = dimensions_movo[0]
MOVO_WIDTH = dimensions_movo[1]
MOVO_HEIGHT = dimensions_movo[2]

front_distance = ((MOVO_LENGTH * 1 / 10) ** 2 + (MOVO_WIDTH / 2) ** 2) ** 0.5
back_distance = ((MOVO_LENGTH * 9 / 10) ** 2 + (MOVO_WIDTH / 2) ** 2) ** 0.5

angle_forward = np.arcsin((MOVO_WIDTH / 2) / front_distance)
angle_backward = np.arcsin((MOVO_WIDTH / 2) / back_distance)


def convert_point_to_pixel(point):

    u, v = point[0], point[2]
    u = int(((u - min_bounds[0]) / dimensions_world[0]) * (dimensions_pixel[0] - 1))
    v = int(((v - min_bounds[1]) / dimensions_world[1]) * (dimensions_pixel[1] - 1))

    return u, v


def pcd_to_grid(pcd, show_camera=False):

    # Just leave space that the robot can collide with
    bb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(min_bounds[0], -1.02, min_bounds[1]),
        max_bound=(max_bounds[0], 0.1, max_bounds[1]),
    )
    pcd = pcd.crop(bb)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=5.0)
    pcd = pcd.select_by_index(ind)

    # o3d.visualization.draw_geometries([pcd])

    # Get the points and compute occupancy grid
    # TRY TO MAKE IT MOPRE EFFICIENT
    points = np.asarray(pcd.points)

    grid_map = np.zeros(dimensions_pixel)

    for point in points:
        u, v = convert_point_to_pixel(point)

        grid_map[u, v] += 1

    grid_map[grid_map <= 20] = 0
    grid_map[grid_map > 20] = 1

    if show_camera:
        f1 = open("KeyFrameTrajectory.txt", "r")
        for line in f1:
            parsed = line.split()
            xyz = np.array([float(parsed[1]), -float(parsed[2]), -float(parsed[3])])

            camera_position = convert_point_to_pixel(xyz)
            grid_map[camera_position] = 5

    grid_map = np.flip(grid_map, axis=0)
    plt.imshow(grid_map)
    plt.show()

    return grid_map


def robot_to_grid(position, theta):
    # r = Rotation.from_quat(quaternion)
    # theta = r.as_rotvec()[1]

    grid = np.zeros(dimensions_pixel)

    points = []
    for i in range(4):
        if i == 0:
            x = front_distance * np.cos(theta + angle_forward)
            y = front_distance * np.sin(theta + angle_forward)

        elif i == 1:
            x = front_distance * np.cos(theta - angle_forward)
            y = front_distance * np.sin(theta - angle_forward)

        elif i == 2:
            x = back_distance * np.cos(theta + angle_backward + np.pi)
            y = back_distance * np.sin(theta + angle_backward + np.pi)

        elif i == 3:
            x = back_distance * np.cos(theta - angle_backward - np.pi)
            y = back_distance * np.sin(theta - angle_backward - np.pi)

        x = -(position[2] + x)
        y = position[0] + y

        points.append(convert_point_to_pixel([x, 0, y]))

    cv2.fillPoly(grid, pts=np.int32([points]), color=1)

    robot_pixels = []
    loc = np.where(grid == 1)

    for pixel in zip(loc[0], loc[1]):
        robot_pixels.append(pixel)

    return robot_pixels


def robot_bb_grid(position, theta):
    grid = np.zeros(dimensions_pixel)

    points = []
    for i in range(4):
        if i == 0:
            x = front_distance * np.cos(theta + angle_forward)
            y = front_distance * np.sin(theta + angle_forward)

        elif i == 1:
            x = front_distance * np.cos(theta - angle_forward)
            y = front_distance * np.sin(theta - angle_forward)

        elif i == 2:
            x = back_distance * np.cos(theta + angle_backward + np.pi)
            y = back_distance * np.sin(theta + angle_backward + np.pi)

        elif i == 3:
            x = back_distance * np.cos(theta - angle_backward - np.pi)
            y = back_distance * np.sin(theta - angle_backward - np.pi)

        x = -(position[2] + x)
        y = position[0] + y

        points.append(convert_point_to_pixel([x, 0, y]))

    cv2.polylines(grid, pts=np.int32([points]), isClosed=True, color=1, thickness=1)

    robot_pixels = []
    loc = np.where(grid == 1)

    for pixel in zip(loc[0], loc[1]):
        robot_pixels.append(pixel)

    return robot_pixels


def expand_grid(grid_map, pcd, camera_pose):

    camera_position = convert_point_to_pixel(
        [camera_pose[0], -camera_pose[1], -camera_pose[2]]
    )

    new_map = np.array(np.zeros(dimensions_pixel))

    bb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(min_bounds[0], -1.02, min_bounds[1]),
        max_bound=(max_bounds[0], 0.1, max_bounds[1]),
    )
    pcd = pcd.crop(bb)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=5.0)
    pcd = pcd.select_by_index(ind)

    points = np.asarray(pcd.points)

    for point in points:
        u, v = convert_point_to_pixel(point)

        new_map[u, v] += 1

    obstacles = np.where(new_map > 20)
    obstacles = list(zip(obstacles[0], obstacles[1]))

    for obstacle in obstacles:
        grid_map[obstacle] = 5
        line = ray_cast(
            camera_position[0], camera_position[1], obstacle[0], obstacle[1]
        )
        for elem in line:
            if grid_map[elem[0], elem[1]] != 5:
                grid_map[elem[0], elem[1]] = 1

    return grid_map


def ray_cast(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    points = []
    while True:
        points.append([x0, y0])
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * error

        if e2 >= dy:
            if x0 == x1:
                break
            error = error + dy
            x0 = x0 + sx
        if e2 <= dx:
            if y0 == y1:
                break
            error = error + dx
            y0 = y0 + sy
    return points


def grid_constructor():

    grid_map = np.array(np.zeros(dimensions_pixel))

    pose_graph = o3d.pipelines.registration.PoseGraph()
    parameters = o3d.camera.PinholeCameraIntrinsic(
        960, 540, 528.612, 531.854, 477.685, 255.955
    )
    f1 = open("KeyFrameTrajectory.txt", "r")
    f2 = open("slam_images.txt", "r")

    pcd = None
    i = 0
    for line in f1:

        line_img = f2.readline().split()

        parsed = line.split()
        xyz = [float(parsed[1]), float(parsed[2]), float(parsed[3])]
        quaternion = [parsed[4], parsed[5], parsed[6], parsed[7]]

        color = o3d.io.read_image(line_img[1])
        depth = o3d.io.read_image(line_img[3])

        new_colors = color
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            new_colors, depth, depth_trunc=1000, convert_rgb_to_intensity=False
        )

        r = Rotation.from_quat(quaternion)
        odom = np.vstack(
            (
                np.hstack((r.as_matrix(), np.array([xyz]).T)),
                np.array([[0.0, 0.0, 0.0, 1]]),
            )
        ).astype(np.float64)

        if i == 0:
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.identity(4))
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, parameters)
            i += 1
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            grid_map = expand_grid(grid_map, pcd, xyz)

            # o3d.visualization.draw_geometries([pcd])
            continue

        else:
            pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, parameters)
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odom))
            pcd_new.transform(pose_graph.nodes[i].pose)
            pcd_new.transform(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            pcd += pcd_new
            grid_map = expand_grid(grid_map, pcd_new, xyz)

        i += 1

    grid_map = np.flip(grid_map, axis=0)
    plt.imshow(grid_map, cmap="gray")
    plt.show()

    return grid_map


if __name__ == "__main__":
    final_grid = grid_constructor()
    plt.imsave("grid.png", final_grid, cmap="gray")
