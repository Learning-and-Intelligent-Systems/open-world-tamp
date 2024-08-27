import os
import time
from itertools import product

import numpy as np
import pybullet as p
import tampura_environments.panda_utils.pb_utils as pbu

MAX_TEXTURE_WIDTH = 418  # max square dimension
MAX_PIXEL_VALUE = 2**8 - 1
MAX_LINKS = 125  # Max links seems to be 126

################################################################################


class VoxelGrid(object):
    def __init__(
        self,
        resolutions,
        default=bool,
        world_from_grid=pbu.unit_pose(),
        aabb=None,
        client=None,
        color=(1, 0, 0, 0.5),
        cloud_vis=False,
        **kwargs,
    ):
        # def __init__(self, sizes, centers, pose=unit_pose()):
        # TODO: defaultdict
        # assert len(sizes) == len(centers)
        assert callable(default)
        self.resolutions = resolutions
        self.default = default
        self.value_from_voxel = {}
        self.world_from_grid = world_from_grid
        self.aabb = aabb  # TODO: apply
        self.color = color
        self.occupied_points = []
        self.occupied_voxel_points = []
        self.cloud_vis = cloud_vis
        self.cloud_handles = []

        # self.bodies = None
        # TODO: store voxels more intelligently spatially

    def clear_cloud(self):
        for cloud_handle in self.cloud_handles:
            pbu.remove_body(cloud_handle)

    def get_frontier(self):
        twod = self.project2d()
        (
            xs,
            ys,
            _,
        ) = zip(*twod)

        twod_array = np.zeros((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
        for i, j, _ in twod:
            twod_array[i - min(xs), j - min(ys)] = 1

        frontiers = []
        for x in range(twod_array.shape[0]):
            for y in range(twod_array.shape[1]):
                frontier = False
                for diffx in [-1, 1]:
                    for diffy in [-1, 1]:
                        if (
                            twod_array[x, y] == 1
                            and x + diffx >= 0
                            and x + diffx < twod_array.shape[0]
                            and y + diffy >= 0
                            and y + diffy < twod_array.shape[1]
                        ):
                            if twod_array[x + diffx, y + diffy] == 0:
                                frontier = True
                if frontier:
                    point = self.pose_from_voxel((x, y, 1))[0]
                    frontiers.append((point[0], point[1]))

        return frontiers

    @property
    def occupied(self):  # TODO: get_occupied
        return sorted(self.value_from_voxel)

    def __iter__(self):
        return iter(self.value_from_voxel)

    def __len__(self):
        return len(self.value_from_voxel)

    def copy(self):  # TODO: deepcopy
        new_grid = VoxelGrid(
            self.resolutions, self.default, self.world_from_grid, self.aabb, self.color
        )
        new_grid.value_from_voxel = dict(self.value_from_voxel)
        return new_grid

    def to_grid(self, point_world):
        return pbu.tform_point(pbu.invert(self.world_from_grid), point_world)

    def to_world(self, point_grid):
        return pbu.tform_point(self.world_from_grid, point_grid)

    def voxel_from_point(self, point):
        point_grid = self.to_grid(point)
        return tuple(np.floor(np.divide(point_grid, self.resolutions)).astype(int))

    # def voxels_from_aabb_grid(self, aabb):
    #    voxel_lower, voxel_upper = map(self.voxel_from_point, aabb)
    #    return map(tuple, product(*[range(l, u + 1) for l, u in safe_zip(voxel_lower, voxel_upper)]))
    def voxels_from_aabb(self, aabb):
        aabb = pbu.aabb_from_points(
            [self.voxel_from_point(point) for point in pbu.get_aabb_vertices(aabb)]
        )
        return map(
            tuple,
            product(
                *[range(l, u + 1) for l, u in pbu.safe_zip(aabb.lower, aabb.upper)]
            ),
        )

    def occupied_voxels_from_aabb(self, aabb):
        vis_points = np.array(self.occupied_points)
        vis_voxels = np.array(self.occupied_voxel_points)
        vis_idx = np.all(
            (aabb.lower <= vis_points) & (vis_points <= aabb.upper), axis=1
        )
        return list([tuple(vp) for vp in vis_voxels[vis_idx]])

    def occupied_voxels_points_from_aabb(self, aabb):
        vis_points = np.array(self.occupied_points)
        vis_idx = np.all(
            (aabb.lower <= vis_points) & (vis_points <= aabb.upper), axis=1
        )
        return vis_points[vis_idx]

    # Grid coordinate frame
    def lower_from_voxel(self, voxel):
        return np.multiply(voxel, self.resolutions)  # self.to_world(

    def center_from_voxel(self, voxel):
        return self.lower_from_voxel(np.array(voxel) + 0.5)

    def upper_from_voxel(self, voxel):
        return self.lower_from_voxel(np.array(voxel) + 1.0)

    def aabb_from_voxel(self, voxel):
        return pbu.AABB(self.lower_from_voxel(voxel), self.upper_from_voxel(voxel))

    def ray_trace(self, start_cell, goal_point):
        # TODO: finish adapting
        if self.is_occupied(start_cell):
            return [], False
        goal_cell = self.get_index(goal_point)
        start_point = self.get_center(start_cell)
        unit = goal_point - start_point
        unit /= np.linalg.norm(unit)
        direction = (unit / np.abs(unit)).astype(int)

        path = []
        current_point = start_point
        current_cell = start_cell
        while current_cell != goal_cell:
            path.append(current_cell)
            min_k, min_t = None, float("inf")
            for k, sign in enumerate(direction):
                next_point = (
                    self.get_min(current_cell)
                    if sign < 0
                    else self.get_max(current_cell)
                )
                t = ((next_point - current_point) / direction)[k]
                assert t > 0
                if (t != 0) and (t < min_t):
                    min_k, min_t = k, t
            assert min_k is not None
            current_point += min_t * unit
            current_cell = np.array(current_cell, dtype=int)
            current_cell[min_k] += direction[min_k]
            current_cell = tuple(current_cell)
            if self.is_occupied(current_cell):
                return path, False
        return path, True

    # World coordinate frame
    def pose_from_voxel(self, voxel):
        pose_grid = pbu.Pose(self.center_from_voxel(voxel))
        return pbu.multiply(self.world_from_grid, pose_grid)

    def vertices_from_voxel(self, voxel):
        return list(
            map(self.to_world, pbu.get_aabb_vertices(self.aabb_from_voxel(voxel)))
        )

    def contains(self, voxel):  # TODO: operator versions
        return voxel in self.value_from_voxel

    def get_value(self, voxel):
        assert self.contains(voxel)
        return self.value_from_voxel[voxel]

    def set_value(self, voxel, value):
        # TODO: remove if value == default
        self.value_from_voxel[voxel] = value

    def remove_value(self, voxel):
        if self.contains(voxel):
            self.value_from_voxel.pop(voxel)  # TODO: return instead?

    is_occupied = contains

    def set_occupied(self, voxel):
        if self.is_occupied(voxel):
            return False
        self.set_value(voxel, value=self.default())
        self.occupied_points.append(list(self.center_from_voxel(voxel)))
        self.occupied_voxel_points.append(voxel)
        return True

    def set_free(self, voxel):
        if not self.is_occupied(voxel):
            return False
        self.remove_value(voxel)
        idx = self.occupied_points.index(list(self.center_from_voxel(voxel)))
        self.occupied_points.remove(self.occupied_points[idx])
        self.occupied_voxel_points.remove(self.occupied_voxel_points[idx])
        return True

    def get_neighbors(self, index):
        for i in range(len(index)):
            direction = np.zeros(len(index), dtype=int)
            for n in (-1, +1):
                direction[i] = n
                yield tuple(np.array(index) + direction)

    def get_clusters(self, voxels=None):
        if voxels is None:
            voxels = self.occupied
        clusters = []
        assigned = set()

        def dfs(current):
            if (current in assigned) or (not self.is_occupied(current)):
                return []
            cluster = [current]
            assigned.add(current)
            for neighbor in self.get_neighbors(current):
                cluster.extend(dfs(neighbor))
            return cluster

        for voxel in voxels:
            cluster = dfs(voxel)
            if cluster:
                clusters.append(cluster)
        return clusters

    # TODO: implicitly check collisions
    def create_box(self, client=None):
        color = (0, 0, 0, 0)
        # color = None
        box = pbu.create_box(*self.resolutions, color=color, client=client)
        # set_color(box, color=color)
        pbu.set_pose(box, self.world_from_grid)  # Set to (0, 0, 0) instead?
        return box

    def get_affected(self, bodies, occupied, client=None):
        check_voxels = {}
        for body in bodies:
            for link in pbu.get_all_links(body, client=client):
                aabb = pbu.get_aabb(
                    body, link, client=client
                )  # TODO: pad using threshold
                for voxel in self.voxels_from_aabb(aabb):
                    if self.is_occupied(voxel) == occupied:
                        check_voxels.setdefault(voxel, []).append((body, link))
        return check_voxels

    def check_collision(self, box, voxel, pairs, threshold=0.0, client=None):
        box_pairs = [(box, link) for link in pbu.get_all_links(box, client=client)]
        pbu.set_pose(box, self.pose_from_voxel(voxel))
        return any(
            pbu.pairwise_link_collision(
                body1, link1, body2, link2, max_distance=threshold, client=client
            )
            for (body1, link1), (body2, link2) in product(pairs, box_pairs)
        )

    def add_point(self, point):
        self.set_occupied(self.voxel_from_point(point))

    def add_aabb(self, aabb):
        for voxel in self.voxels_from_aabb(aabb):
            self.set_occupied(voxel)

    def add_body(self, body, **kwargs):
        self.add_bodies([body], **kwargs)

    def add_bodies(self, bodies, threshold=0.0, client=None):
        # Otherwise, need to transform bodies
        check_voxels = self.get_affected(bodies, occupied=False)
        box = self.create_box()
        for (
            voxel,
            pairs,
        ) in check_voxels.items():  # pairs typically only has one element
            if self.check_collision(box, voxel, pairs, threshold=threshold):
                self.set_occupied(voxel)
        pbu.remove_body(box, client=client)

    def remove_body(self, body, **kwargs):
        self.remove_bodies([body], **kwargs)

    def remove_bodies(self, bodies, client=None, **kwargs):
        # TODO: could also just iterate over the voxels directly
        check_voxels = self.get_affected(bodies, occupied=True)
        box = self.create_box()
        for voxel, pairs in check_voxels.items():
            if self.check_collision(box, voxel, pairs, **kwargs):
                self.set_free(voxel)
        pbu.remove_body(box, client=client)

    def draw_origin(self, scale=1, client=None, **kwargs):
        size = scale * np.min(self.resolutions)
        return pbu.draw_pose(self.world_from_grid, length=size, client=client, **kwargs)

    def draw_voxel(self, voxel, color=None, client=None):
        if color is None:
            color = self.color
        aabb = self.aabb_from_voxel(voxel)
        return pbu.draw_oobb(
            pbu.OOBB(aabb, self.world_from_grid), color=color[:3], client=client
        )
        # handles.extend(draw_aabb(aabb, color=self.color[:3]))

    def draw_voxel_boxes(self, voxels=None, client=None, **kwargs):
        if voxels is None:
            voxels = self.occupied
        with pbu.LockRenderer(client=client):
            handles = []
            for voxel in voxels:
                handles.extend(self.draw_voxel(voxel, **kwargs))
            return handles

    def draw_voxel_centers(self, voxels=None, color=None, client=None):
        # TODO: could align with grid orientation
        if voxels is None:
            voxels = self.occupied
        if color is None:
            color = self.color
        with pbu.LockRenderer(client=client):
            size = np.min(self.resolutions) / 2
            handles = []
            for voxel in voxels:
                point_world = self.to_world(self.center_from_voxel(voxel))
                handles.extend(
                    pbu.draw_point(
                        point_world, size=size, color=color[:3], client=client
                    )
                )
            return handles

    def create_voxel_bodies1(self, client=None):
        start_time = time.time()
        geometry = pbu.get_box_geometry(*self.resolutions, client=client)
        collision_id, visual_id = pbu.create_shape(
            geometry, color=self.color, client=client
        )
        bodies = []
        for voxel in self.occupied:
            body = pbu.create_body(collision_id, visual_id, client=client)
            # scale = self.resolutions[0]
            # body = load_model('models/voxel.urdf', fixed_base=True, scale=scale)
            pbu.set_pose(body, self.pose_from_voxel(voxel))
            bodies.append(body)  # 0.0462474774444 / voxel
        return bodies

    def create_voxel_bodies2(self, client=None):
        geometry = pbu.get_box_geometry(*self.resolutions, client=client)
        collision_id, visual_id = pbu.create_shape(
            geometry, color=self.color, client=client
        )
        ordered_voxels = self.occupied
        bodies = []
        for start in range(0, len(ordered_voxels), MAX_LINKS):
            voxels = ordered_voxels[start : start + MAX_LINKS]
            body = client.createMultiBody(
                linkMasses=len(voxels) * [pbu.STATIC_MASS],
                linkCollisionShapeIndices=len(voxels) * [collision_id],
                linkVisualShapeIndices=len(voxels) * [visual_id],
                linkPositions=list(map(self.center_from_voxel, voxels)),
                linkOrientations=len(voxels) * [pbu.unit_quat()],
                linkInertialFramePositions=len(voxels) * [pbu.unit_point()],
                linkInertialFrameOrientations=len(voxels) * [pbu.unit_quat()],
                linkParentIndices=len(voxels) * [0],
                linkJointTypes=len(voxels) * [p.JOINT_FIXED],
                linkJointAxis=len(voxels) * [pbu.unit_point()],
                physicsClientId=pbu.CLIENT,
            )
            pbu.set_pose(body, self.world_from_grid)
            bodies.append(body)  # 0.0163199263677 / voxel
        return bodies

    def create_voxel_bodies3(self, client=None):
        ordered_voxels = self.occupied
        geoms = [
            pbu.get_box_geometry(*self.resolutions, client=client)
            for _ in ordered_voxels
        ]
        poses = list(map(self.pose_from_voxel, ordered_voxels))
        # colors = [list(self.color) for _ in self.voxels] # TODO: colors don't work
        colors = None
        collision_id, visual_id = pbu.create_shape_array(
            geoms, poses, colors, client=client
        )
        body = pbu.create_body(
            collision_id, visual_id, client=client
        )  # Max seems to be 16
        # dump_body(body)
        pbu.set_color(body, self.color)
        return [body]

    def create_voxel_bodies(self, client=None):
        with pbu.LockRenderer(client=client):
            return self.create_voxel_bodies1()
            # return self.create_voxel_bodies2()
            # return self.create_voxel_bodies3()

    def create_intervals(self):
        voxel_heights = {}
        for i, j, k in self.occupied:
            voxel_heights.setdefault((i, j), set()).add(k)
        voxel_intervals = []
        for i, j in voxel_heights:
            heights = sorted(voxel_heights[i, j])
            start = last = heights[0]
            for k in heights[1:]:
                if k == last + 1:
                    last = k
                else:
                    interval = (start, last)
                    voxel_intervals.append((i, j, interval))
                    start = last = k
            interval = (start, last)
            voxel_intervals.append((i, j, interval))

        return voxel_intervals

    def draw_intervals(self, client=None):
        with pbu.LockRenderer(client=client):
            handles = []
            for i, j, (k1, k2) in self.create_intervals():
                voxels = [(i, j, k1), (i, j, k2)]
                aabb = pbu.aabb_from_points(
                    [
                        extrema
                        for voxel in voxels
                        for extrema in [
                            self.aabb_from_voxel(voxel).lower,
                            self.aabb_from_voxel(voxel).upper,
                        ]
                    ]
                )
                if self.cloud_vis:
                    self.cloud_handles.extend(
                        draw_cloud_oobb(
                            pbu.OOBB(aabb, self.world_from_grid),
                            color=[self.color.red, self.color.green, self.color.blue],
                            client=client,
                        )
                    )
                else:
                    handles.extend(
                        pbu.draw_oobb(
                            pbu.OOBB(aabb, self.world_from_grid),
                            color=[self.color.red, self.color.green, self.color.blue],
                            client=client,
                        )
                    )
            return handles

    def draw_vertical_lines(self, client=None):
        with pbu.LockRenderer(client=client):
            handles = []
            for i, j, (k1, k2) in self.create_intervals():
                voxels = [(i, j, k1), (i, j, k2)]
                aabb = pbu.aabb_from_points(
                    [
                        extrema
                        for voxel in voxels
                        for extrema in [
                            self.aabb_from_voxel(voxel).lower,
                            self.aabb_from_voxel(voxel).upper,
                        ]
                    ]
                )
                center = pbu.get_aabb_center(aabb)
                p1 = self.to_world(np.append(center[:2], [aabb[0][2]]))
                p2 = self.to_world(np.append(center[:2], [aabb[1][2]]))
                handles.append(
                    pbu.add_line(p1, p2, color=self.color[:3], client=client)
                )
            return handles

    def project2d(self):
        # TODO: combine adjacent voxels into larger lines
        # TODO: greedy algorithm that combines lines/boxes
        # TODO: combine intervals
        tallest_voxel = {}
        for i, j, k in self.occupied:
            tallest_voxel[i, j] = max(k, tallest_voxel.get((i, j), k))
        return {(i, j, k) for (i, j), k in tallest_voxel.items()}

    def create_height_map(
        self,
        plane,
        plane_size,
        width=MAX_TEXTURE_WIDTH,
        height=MAX_TEXTURE_WIDTH,
        client=None,
    ):
        min_z, max_z = 0.0, 2.0
        plane_extent = plane_size * np.array([1, 1, 0])
        plane_lower = pbu.get_point(plane, client=client) - plane_extent / 2.0
        # plane_aabb = (plane_lower, plane_lower + plane_extent)
        # plane_aabb = get_aabb(plane) # TODO: bounding box is effectively empty
        # plane_lower, plane_upper = plane_aabb
        # plane_extent = (plane_upper - plane_lower)
        image_size = np.array([width, height])
        # TODO: fix width/height order
        pixel_from_point = lambda point: np.floor(
            image_size * (point - plane_lower)[:2] / plane_extent[:2]
        ).astype(int)

        # TODO: last row/col doesn't seem to be filled
        height_map = np.zeros(image_size)
        for voxel in self.project2d():
            voxel_aabb = self.aabb_from_voxel(voxel)
            # if not aabb_contains_aabb(aabb2d_from_aabb(voxel_aabb), aabb2d_from_aabb(plane_aabb)):
            #    continue
            (x1, y1), (x2, y2) = map(pixel_from_point, voxel_aabb)
            if (x1 < 0) or (width <= x2) or (y1 < 0) or (height <= y2):
                continue
            scaled_z = (pbu.clip(voxel_aabb[1][2], min_z, max_z) - min_z) / max_z
            for c in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    r = (
                        height - y - 1
                    )  # TODO: can also just set in bulk if using height_map
                    height_map[r, c] = max(height_map[r, c], scaled_z)
        return height_map


def draw_cloud_oobb(oobb: pbu.OOBB, color=None, **kwargs):
    CLOUD_HEIGHT = 1.5
    color = pbu.RGBA(color[0], color[1], color[2], 0.1)
    collision_id, visual_id = pbu.create_shape(
        pbu.get_box_geometry(
            oobb.aabb.upper[0] - oobb.aabb.lower[0],
            oobb.aabb.upper[1] - oobb.aabb.lower[1],
            CLOUD_HEIGHT,
        ),
        color=color,
    )
    body = pbu.create_body(collision_id, visual_id, mass=0, **kwargs)
    pbu.set_pose(
        body,
        pbu.Pose(
            [
                (oobb.aabb.upper[0] + oobb.aabb.lower[0]) / 2.0,
                (oobb.aabb.upper[1] + oobb.aabb.lower[1]) / 2.0,
                CLOUD_HEIGHT / 2.0,
            ],
            [0, 0, 0],
        ),
    )
    return [body]


def rgb_interpolate(grey_image, min_color, max_color):
    width, height = grey_image.shape
    channels = 3
    rgb_image = np.zeros((width, height, channels), dtype=np.uint8)
    for k in range(channels):
        rgb_image[..., k] = MAX_PIXEL_VALUE * (
            min_color[k] * (1 - grey_image) + max_color[k] * grey_image
        )
    return rgb_image
