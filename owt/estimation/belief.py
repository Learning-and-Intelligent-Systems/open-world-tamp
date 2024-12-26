import time
from collections import namedtuple

import numpy as np

import owt.pb_utils as pbu
from owt.estimation.clustering import relabel_clusters
from owt.estimation.geometry import estimate_surface_mesh
from owt.estimation.observation import (LabeledPoint, aggregate_color,
                                        custom_iterate_point_cloud,
                                        iterate_image, iterate_point_cloud,
                                        save_image_from_cluster,
                                        tform_labeled_points)
from owt.estimation.surfaces import plane_from_pose, surface_point_filter
from owt.simulation.entities import (ENVIRONMENT, TABLE, UNKNOWN, Object,
                                     displace_body)
from owt.voxel_utils import VoxelGrid

GRASP_EXPERIMENT = True

if GRASP_EXPERIMENT:
    FRAGILE_HEIGHT = 10e-2  # Tall fragile
else:
    FRAGILE_HEIGHT = 0.0  # All fragile

ObjectGrid = namedtuple("ObjectGrid", ["voxelgrid", "category"])


def is_object_label(label):
    category, instance = label
    return (category not in ENVIRONMENT) and isinstance(instance, str)


def is_surface(cluster, threshold=0.005):
    category, _ = cluster[0].label
    pt = [lp.point[2] for lp in cluster]
    return (category == TABLE) or (max(pt) - min(pt) <= threshold)


################################################################################


class EstimatedObject(Object):
    def __init__(
        self,
        body,
        color=None,
        labeled_points=[],
        voxels=frozenset(),
        unknown=frozenset(),
        contains=[],
        is_fragile=True,
        **kwargs
    ):
        super(EstimatedObject, self).__init__(body, **kwargs)
        # TODO: could also just set the transparency
        self.labeled_points = tuple(labeled_points)
        self.contains = contains
        self.voxels = frozenset(voxels)
        self.unknown = frozenset(unknown)
        self.is_fragile = is_fragile

        # TODO: store OOBB
        if (color is None) and self.labeled_points:
            # hue = mean_hue(self.colors, min_sat=0.5, min_value=0.5)
            # if hue is None:
            color = aggregate_color(self.labeled_points)
            # else:
            #    color = apply_alpha(np.array(colorsys.hsv_to_rgb(h=hue, s=1., v=1.)))
            pbu.set_all_color(
                self.body, pbu.apply_alpha(color, alpha=1.0), client=self.client
            )
        self.color = color  # TODO: could extract from the body/texture instead

    def update(self, newcloud, tform, camera_image, min_points, surface, **kwargs):
        tformed_points = tform_labeled_points(
            pbu.pose_from_tform(tform), self.labeled_points
        )
        # TODO take closest point?
        label = self.labeled_points[0].label  # TODO asso
        get_color = lambda pt: camera_image[0][
            tuple(
                np.floor(
                    pbu.pixel_from_ray(
                        camera_image[4],
                        pbu.tform_point(pbu.invert(camera_image[3]), pt),
                    )
                )[::-1].astype(int)
            )
        ]
        # import pdb;pdb.set_trace()
        new_pt = [
            LabeledPoint(point, get_color(point), label) for point in newcloud
        ]  # TODO asso
        random_shuffle_pt = downsample_cluster(new_pt + tformed_points)
        maxz = pbu.get_aabb(surface.body).upper[
            2
        ]  # TODO how about estimated surface objs?
        random_shuffle_pt = [
            lp for lp in random_shuffle_pt if lp.point[2] - 1e-2 >= maxz
        ]
        # relevant_cloud = surface_point_filter(surface, random_shuffle_pt)
        clusters = relabel_clusters(random_shuffle_pt)
        # TODO: relative to the robot
        # TODO loop over all clusters?
        cluster = sorted(clusters, key=lambda c: len(c), reverse=True)[0]
        if len(cluster) < min_points:
            import pdb

            pdb.set_trace()
        body = estimate_surface_mesh(cluster, camera_image=camera_image, **kwargs)
        if body is None:
            self.remove()
            return False
        # remove old body
        self.remove()

        self.labeled_points = cluster
        self.body = body
        color = aggregate_color(self.labeled_points)
        pbu.set_all_color(self.body, pbu.apply_alpha(color / 255.0, alpha=1.0))
        self.color = color  # TODO: could extract from the body/texture instead
        self.name = "{}#{}".format(self.category, self.body)
        return True

    @property
    def colors(self):
        return [lp.color for lp in self.labeled_points]

    # @property
    # def color(self):
    #    return aggregate_color(self.labeled_points)


################################################################################


class SurfaceBelief(Object):
    def __init__(
        self,
        surface,
        height=0.25,
        resolutions=0.05 * np.ones(3),
        known_objects=[],
        client=None,
        **kwargs
    ):
        # TODO: be careful about the cube created
        assert len(resolutions) == 3
        self.surface = surface
        self.client = client
        self.surface_pose = pbu.get_pose(surface)
        self.height = height
        self.resolutions = resolutions
        self.known_objects = frozenset(known_objects) | {self.surface}
        (min_x, min_y, _), (max_x, max_y, max_z) = pbu.get_aabb(
            surface
        )  # TODO: pybullet buffers by about 2e-2
        self.surface_aabb = pbu.AABB(
            lower=(min_x, min_y, max_z), upper=(max_x, max_y, max_z + height)
        )  # TODO: buffer
        self.handles = []

        self.num_updates = 0  # TODO: store the observation history
        self.observations = []  # TODO: store a pointer to the parent belief
        self.reset()
        # TODO: adjust orientation

    @property
    def surface_height(self):
        return self.surface_aabb[0][2]

    @property
    def surface_origin(self):
        return pbu.Pose(pbu.Point(z=self.surface_height))  # TODO: orientation

    @property
    def surface_plane(self):
        # create_rectangular_surface
        return plane_from_pose(self.surface_origin)

    def create_grid(self, **kwargs):
        return VoxelGrid(
            self.resolutions,
            world_from_grid=self.surface_origin,
            aabb=self.surface_aabb,
            **kwargs
        )

    def reset_objects(self):
        for estimate in self.estimated_objects:
            estimate.remove()
        self.estimated_objects = []

    def reset(self):
        # TODO: selectively reset or decrease confidence
        self.estimated_objects = []
        self.occupancy_grid = self.create_grid(color=pbu.RED)  # TODO: just use unknown
        self.visibility_grid = self.create_grid(color=pbu.BLUE)
        for voxel in self.visibility_grid.voxels_from_aabb(self.surface_aabb):
            self.visibility_grid.set_occupied(voxel)
        self.class_grids = {}

    def erase(self):
        pbu.remove_handles(self.handles)
        self.handles = []

    def update_occupancy(self, camera_image, **kwargs):
        # TODO: is_object_point(lp)
        relevant_cloud = [
            lp
            for lp in iterate_point_cloud(camera_image, **kwargs)
            if is_object_label(lp.label)
            and pbu.aabb_contains_point(lp.point, self.surface_aabb)
        ]  # TODO replace with matrix operation
        relevant_cloud = list(pbu.flatten(relabel_clusters(relevant_cloud)))
        for labeled_point in relevant_cloud:
            point = labeled_point.point
            category, instance = labeled_point.label
            # if is_object_point(labeled_point):
            if isinstance(instance, str) and (
                category != UNKNOWN
            ):  # TODO: could do for an Object as well
                # TODO: data association of label[1]
                if instance not in self.class_grids:
                    self.class_grids[instance] = ObjectGrid(
                        self.create_grid(default=list, color=pbu.BLUE), category
                    )  # TODO: color
                voxel = self.class_grids[instance].voxelgrid.voxel_from_point(point)
                self.class_grids[instance].voxelgrid.add_point(point)
                self.class_grids[instance].voxelgrid.get_value(voxel).append(
                    labeled_point
                )
            else:
                # TODO: the different is really just whether we'll make a mesh or not
                self.occupancy_grid.add_point(point)

    def update_visibility(self, camera_image, **kwargs):
        camera_pose, camera_matrix = camera_image[-2:]
        grid = self.visibility_grid
        for voxel in grid.voxels_from_aabb(self.surface_aabb):
            center_world = grid.to_world(grid.center_from_voxel(voxel))
            center_camera = pbu.tform_point(pbu.invert(camera_pose), center_world)
            distance = center_camera[2]
            pixel = pbu.pixel_from_point(camera_matrix, center_camera)
            if pixel is not None:
                r, c = pixel.row, pixel.column
                depth = camera_image.depthPixels[r, c]
                if distance <= depth:
                    grid.set_free(voxel)

    def update(self, camera_image, visibility=True):
        start_time = time.time()
        self.num_updates += 1
        self.observations.append(camera_image)
        self.update_occupancy(camera_image)
        if visibility:
            self.update_visibility(camera_image)
        # self.estimate_bodies()
        print("Update time: {:.3f}".format(pbu.elapsed_time(start_time)))

    def estimate_bodies(self, min_voxels=1, min_points=3, **kwargs):
        self.reset_objects()
        for instance_name, (grid, category) in self.class_grids.items():
            for i, cluster in enumerate(grid.get_clusters()):
                if len(cluster) < min_voxels:
                    continue
                # name = '~{}#{}'.format(class_name, i+1)
                name = instance_name
                # name = None
                labeled_points = list(
                    pbu.flatten(grid.get_value(voxel) for voxel in cluster)
                )
                if len(labeled_points) < min_points:
                    continue
                color = aggregate_color(labeled_points)
                body = estimate_surface_mesh(
                    labeled_points,
                    self.surface_origin,
                    camera_image=self.observations[-1],
                    client=self.client,
                    **kwargs
                )
                if body is None:
                    continue

                adjacent = set(
                    pbu.flatten(grid.get_neighbors(voxel) for voxel in cluster)
                ) - set(cluster)
                unknown = {
                    voxel for voxel in adjacent if self.visibility_grid.contains(voxel)
                }

                self.estimated_objects.append(
                    EstimatedObject(
                        body,
                        name=name,
                        category=category,
                        color=color,
                        labeled_points=labeled_points,
                        voxels=cluster,
                        unknown=unknown,
                    )
                )
                # grid.draw_voxel_boxes(cluster)
                # wait_if_gui()
        # TODO: convex hull of occupied vertices
        return self.estimated_objects

    def fuse_occupancy(self):  # Belief consistency / revision
        estimated_bodies = [obj.body for obj in self.estimated_objects]
        filtered_occupancy = self.occupancy_grid.copy()
        filtered_occupancy.remove_bodies(estimated_bodies)
        return filtered_occupancy

    def fuse_visibility(self):  # Belief consistency / revision
        estimated_bodies = [obj.body for obj in self.estimated_objects]
        filtered_visibility = self.visibility_grid.copy()
        # for voxel in self.occupancy_grid:
        for voxel in self.fuse_occupancy():
            filtered_visibility.set_free(voxel)
        filtered_visibility.remove_bodies(estimated_bodies)
        return filtered_visibility

    def fuse(self):
        SurfaceBelief(self.surface, self.height, self.resolutions, self.known_objects)
        raise NotImplementedError()
        # return filtered_belief

    def draw(self, **kwargs):
        start_time = time.time()
        self.erase()

        self.handles = list(
            pbu.flatten(
                [
                    pbu.draw_pose(self.surface_origin),
                    pbu.draw_aabb(self.surface_aabb, **kwargs),
                    self.fuse_occupancy().draw_intervals(),  # draw_voxel_boxes
                    self.fuse_visibility().draw_vertical_lines(),
                ]
            )
        )
        # for grid in self.class_grids.values():
        #    self.handles.extend(grid.draw_voxel_boxes())

        print("Draw time: {:.3f}".format(pbu.elapsed_time(start_time)))
        return self.handles


class Belief(object):
    def __init__(
        self,
        robot,
        surface_beliefs=[],
        known_objects=[],
        known_surfaces=[],
        materials={},
        displacement=10,
        client=None,
        **kwargs
    ):
        self.robot = robot
        self.materials = materials
        self.client = client
        self.surface_beliefs = tuple(surface_beliefs)  # dict
        self.known_objects = tuple(known_objects)
        self.known_surfaces = tuple(known_surfaces)  # TODO: unify with surface_beliefs
        self.observations = []
        self.estimated_objects = []

        self.displacement = displacement
        self.saver = None

    @property
    def surfaces(self):
        return [surface_belief.surface for surface_belief in self.surface_beliefs]

    @property
    def obstacles(self):
        return [
            body for body in self.known_objects if body != self.robot
        ]  # TODO: known_objects?

    def __repr__(self):
        # TODO: dump
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            self.robot,
            self.known_surfaces,
            self.estimated_objects,
        )

    def add_surface(self, surface):
        raise NotImplementedError()

    def reset_objects(self):
        for estimate in self.estimated_objects:
            estimate.remove()
        self.estimated_objects = []

    def reset_surfaces(self):
        for estimate in self.known_surfaces:
            estimate.remove()
        self.known_surfaces = []

    def reset(self):
        self.reset_objects()
        self.reset_surfaces()
        for surface_belief in self.surface_beliefs:
            surface_belief.reset()

    def estimate_objects(
        self,
        camera_image,
        use_seg=True,
        surface=None,
        min_points=30,
        save_relabeled_img=False,
        surfaces_movable=True,
        max_depth=float("inf"),
        **kwargs
    ):
        # TODO: make a standalone method
        self.reset_objects()
        self.observations.append(camera_image)

        # st = time.time()
        pixels = [
            pixel
            for pixel in iterate_image(camera_image, step_size=2)
            if is_object_label(camera_image.segmentationMaskBuffer[pixel])
        ]

        relevant_cloud = [
            lp
            for lp in custom_iterate_point_cloud(
                camera_image, pixels, max_depth=max_depth
            )
        ]

        if surface is not None:  # TODO speedup todo
            relevant_cloud = surface_point_filter(
                surface, relevant_cloud
            )  # TODO: keep any point above the surface (for unlabeled)
        # TODO: prune close to the table here not earlier
        # TODO: labeled unlabeled points and then cluster to pick up on objects missed by the segmentation
        if not use_seg:  # USING_ROS:
            clusters = relabel_clusters(
                relevant_cloud,
                use_instance_label=False,
                use_trimesh=True,
                radius=3e-2,
                use_2d=True,
                noise_only=False,
            )
        else:
            clusters = relabel_clusters(relevant_cloud)
        clusters = sorted(
            clusters, key=lambda c: min(lp.point[0] for lp in c)
        )  # TODO: relative to the robot
        for cluster in clusters:
            if len(cluster) < min_points:
                continue
            category, bid = cluster[0].label
            min_z = min(lp.point[2] for lp in cluster)
            max_z = max(lp.point[2] for lp in cluster)
            surface_pose = pbu.Pose(
                pbu.Point(z=min_z)
            )  # if surface is None else surface.pose
            body = estimate_surface_mesh(
                downsample_cluster(cluster),
                surface_pose,
                camera_image=camera_image,
                min_points=min_points,
                client=self.client,
                **kwargs
            )
            if body is None:
                continue

            # surfaces_movable decides whether or not to treat surfaces as movable objects
            if surfaces_movable or not is_surface(cluster):
                height = max_z - min_z
                is_fragile = height > FRAGILE_HEIGHT
                self.estimated_objects.append(
                    EstimatedObject(
                        body,
                        category=category,
                        labeled_points=cluster,
                        is_fragile=is_fragile,
                        client=self.client,
                    )
                )
        if save_relabeled_img:
            save_image_from_cluster(clusters, camera_image, min_points=min_points)
        # clusters = [body.labeled_points for body in self.estimated_objects]
        # save_image_from_cluster(clusters, camera_image, min_points=min_points, imname=f'relabeled_proj_{len(self.observations)-1}.png')
        return self.estimated_objects

    def relabel_color(self, assumption_color):
        import colorsys

        rgb_anchor_hue = [0.0, 1 / 3.0, 2 / 3.0]
        name_to_class = {"red": 0, "green": 1, "blue": 2}
        class_to_name = {v: k for (k, v) in name_to_class.items()}

        colors_estimated_hue = [
            colorsys.rgb_to_hsv(*obj.color[:3])[0] for obj in self.estimated_objects
        ]  # K
        color_dist = lambda x, y: min(abs(x - y), 1 - abs(x - y))
        classification_matrix = np.array(
            [
                [np.exp(-color_dist(c_pred_hue, chr_hue)) for chr_hue in rgb_anchor_hue]
                for c_pred_hue in colors_estimated_hue
            ]
        )  # K(pred) x 3
        relabeled_classes = take_ml_estimate(
            classification_matrix, [name_to_class[c] for c in assumption_color]
        )
        for i in range(len(self.estimated_objects)):
            self.estimated_objects[i].color = pbu.apply_alpha(
                pbu.COLOR_FROM_NAME[class_to_name[relabeled_classes[i]]]
            )
        return self.estimated_objects

    def disable(self):
        if self.saver is None:
            self.saver = pbu.WorldSaver(bodies=self.estimated_objects)
            for body in self.estimated_objects:
                vector = self.displacement * pbu.get_unit_vector([1, 0, 0])
                displace_body(body, vector)
        return self.saver

    def enable(self):
        if self.saver is not None:
            self.saver.restore()
            self.saver = None
        return self.saver


def take_ml_estimate(prob_matrix, target_classes):
    original_classes = np.argmax(prob_matrix, 1)
    relabeled_classes = original_classes.copy()
    not_relabeled = np.ones(prob_matrix.shape[0])
    for target_class in target_classes:
        remain = not_relabeled.nonzero()[0]
        if len(remain) == 0:
            break  # failed
        ml_id = np.argmax(prob_matrix[remain][:, target_class])
        relabeled_classes[remain[ml_id]] = target_class
        not_relabeled[remain[ml_id]] = 0
    return relabeled_classes


def get_llist(lista, listb):
    assert len(lista) == len(listb)
    listlen = len(lista)
    for i in range(listlen):
        yield (lista[i], listb[i])


def downsample_cluster(cluster, num_pts=1000):
    if len(cluster) <= num_pts:
        return cluster
    else:
        # c_mask = np.zeros(len(cluster), dtype=int)
        # c_mask[:num_pts] = 1
        np.random.shuffle(cluster)
        return cluster[:num_pts]
        # return cluster[c_mask.nonzero()]
