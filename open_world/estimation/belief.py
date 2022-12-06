import time
from collections import namedtuple

import numpy as np
from pybullet_tools.utils import (
    AABB,
    BLUE,
    COLOR_FROM_NAME,
    RED,
    Point,
    Pose,
    WorldSaver,
    aabb_contains_point,
    apply_alpha,
    draw_aabb,
    draw_pose,
    elapsed_time,
    flatten,
    get_aabb,
    get_pose,
    get_unit_vector,
    invert,
    pixel_from_point,
    pixel_from_ray,
    pose_from_tform,
    remove_handles,
    set_all_color,
    tform_from_pose,
    tform_point,
)
from pybullet_tools.voxels import VoxelGrid
from scipy.optimize import linear_sum_assignment

from open_world.estimation.clustering import relabel_clusters
from open_world.estimation.geometry import estimate_surface_mesh
from open_world.estimation.observation import (
    LabeledPoint,
    aggregate_color,
    custom_iterate_point_cloud,
    iterate_image,
    iterate_point_cloud,
    save_image_from_cluster,
    tform_labeled_points,
)
from open_world.estimation.surfaces import plane_from_pose, surface_point_filter
from open_world.simulation.entities import (
    ENVIRONMENT,
    TABLE,
    UNKNOWN,
    Object,
    displace_body,
)

# from open_world.estimation.association import compute_tform, get_overlap_obj, get_overlap_objcloud


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
            set_all_color(self.body, apply_alpha(color, alpha=1.0), client=self.client)
        self.color = color  # TODO: could extract from the body/texture instead

    def update(self, newcloud, tform, camera_image, min_points, surface, **kwargs):
        tformed_points = tform_labeled_points(
            pose_from_tform(tform), self.labeled_points
        )
        # TODO take closest point?
        label = self.labeled_points[0].label  # TODO asso
        get_color = lambda pt: camera_image[0][
            tuple(
                np.floor(
                    pixel_from_ray(
                        camera_image[4], tform_point(invert(camera_image[3]), pt)
                    )
                )[::-1].astype(int)
            )
        ]
        # import pdb;pdb.set_trace()
        new_pt = [
            LabeledPoint(point, get_color(point), label) for point in newcloud
        ]  # TODO asso
        random_shuffle_pt = downsample_cluster(new_pt + tformed_points)
        maxz = get_aabb(surface.body).upper[2]  # TODO how about estimated surface objs?
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
        set_all_color(self.body, apply_alpha(color / 255.0, alpha=1.0))
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
        self.surface_pose = get_pose(surface)
        self.height = height
        self.resolutions = resolutions
        self.known_objects = frozenset(known_objects) | {self.surface}
        (min_x, min_y, _), (max_x, max_y, max_z) = get_aabb(
            surface
        )  # TODO: pybullet buffers by about 2e-2
        self.surface_aabb = AABB(
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
        return Pose(Point(z=self.surface_height))  # TODO: orientation

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
        self.occupancy_grid = self.create_grid(color=RED)  # TODO: just use unknown
        self.visibility_grid = self.create_grid(color=BLUE)
        for voxel in self.visibility_grid.voxels_from_aabb(self.surface_aabb):
            self.visibility_grid.set_occupied(voxel)
        self.class_grids = {}

    def erase(self):
        remove_handles(self.handles)
        self.handles = []

    def update_occupancy(self, camera_image, **kwargs):
        # TODO: is_object_point(lp)
        relevant_cloud = [
            lp
            for lp in iterate_point_cloud(camera_image, **kwargs)
            if is_object_label(lp.label)
            and aabb_contains_point(lp.point, self.surface_aabb)
        ]  # TODO replace with matrix operation
        relevant_cloud = list(flatten(relabel_clusters(relevant_cloud)))
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
                        self.create_grid(default=list, color=BLUE), category
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
            center_camera = tform_point(invert(camera_pose), center_world)
            distance = center_camera[2]
            # if not (0 <= distance < max_depth):
            #    continue
            pixel = pixel_from_point(camera_matrix, center_camera)
            if pixel is not None:
                # TODO: local filter
                r, c = pixel
                depth = camera_image.depthPixels[r, c]
                # if distance > depth:
                #     grid.set_occupied(voxel)
                #     # grid.add_point(center_world) # TODO: check pixels within voxel bounding box
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
        print("Update time: {:.3f}".format(elapsed_time(start_time)))

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
                    flatten(grid.get_value(voxel) for voxel in cluster)
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
                    flatten(grid.get_neighbors(voxel) for voxel in cluster)
                ) - set(cluster)
                unknown = {
                    voxel for voxel in adjacent if self.visibility_grid.contains(voxel)
                }  # TODO: use the mesh
                # print(name, len(cluster), len(labeled_points), len(adjacent), len(unknown))

                # grid.draw_voxel_boxes(cluster, color=RED)
                # #grid.draw_voxel_boxes(adjacent)
                # grid.draw_voxel_boxes(unknown)
                # wait_if_gui()
                
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
        # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/044d5b22f6976f7d1c7ba53cc445ce01a557646e/plan_tools/retired/debug.py#L23
        start_time = time.time()
        self.erase()
        # grid.draw_voxel_centers()
        # grid.draw_voxel_boxes()
        # grid.draw_intervals()
        # for voxel in grid.voxels_from_aabb(surface_aabb):
        #    draw_point(grid.to_world(grid.center_from_voxel(voxel)))

        # with LockRenderer():
        #    grid.add_body(obj1)

        self.handles = list(
            flatten(
                [
                    draw_pose(self.surface_origin),
                    draw_aabb(self.surface_aabb, **kwargs),
                    self.fuse_occupancy().draw_intervals(),  # draw_voxel_boxes
                    self.fuse_visibility().draw_vertical_lines(),
                ]
            )
        )
        # for grid in self.class_grids.values():
        #    self.handles.extend(grid.draw_voxel_boxes())

        print("Draw time: {:.3f}".format(elapsed_time(start_time)))
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

    def update(
        self,
        camera_image_obs,
        camera_image_pred,
        tform_poses,
        flow_network=None,
        use_seg=True,
        surface=None,
        min_points=30,
        save_relabeled_img=False,
        surfaces_movable=True,
        max_depth=float("inf"),
        **kwargs
    ):
        camera_image_prev = self.observations[-1]
        self.observations.append(camera_image_obs)
        _, dep_obs, seg_obs, _, _ = camera_image_obs
        _, dep_pred, seg_pred, _, _ = camera_image_pred
        _, dep_prev, seg_prev, _, _ = camera_image_prev

        scene_diff = dep_obs - dep_pred  # TODO noise in real world
        diff_thres = 3e-2
        scene_diff_validmask = (scene_diff ** 2) ** 0.5 >= diff_thres
        scene_diff[~scene_diff_validmask] = 0
        disappeared = scene_diff > 0
        appeared = scene_diff < 0
        # plt.subplot(131);plt.imshow(disappeared);plt.subplot(132);plt.imshow(appeared);plt.subplot(133);plt.imshow(scene_diff_validmask);plt.show();plt.close()
        # plt.subplot(131);plt.imshow(dep_obs);plt.subplot(132);plt.imshow(dep_pred);plt.subplot(133);plt.imshow(scene_diff_validmask);plt.show();plt.close()

        all_projmask, removed_objs = get_overlap_obj(
            disappeared, camera_image_pred, self.estimated_objects, tform_poses
        )  # get the obj ids
        # TODO asso. use occupancy grid
        # TODO moving camera
        obs_cloud, added_objs, added_objs_segname = get_overlap_objcloud(
            appeared, camera_image_obs
        )
        for obj in self.estimated_objects:
            if obj not in removed_objs:
                tform_planned = tform_from_pose(tform_poses[obj])
                update_success = obj.update(
                    [],
                    tform_planned,
                    camera_image_obs,
                    surface=surface,
                    min_points=30,
                    **kwargs
                )
                if not update_success:
                    removed_objs.append(obj)

        asso_thres = 0.01
        prev_to_cur_tform = np.empty((len(removed_objs), len(added_objs), 4, 4))
        prev_to_cur_dist = np.empty((len(removed_objs), len(added_objs)))
        for i, prev_obj in enumerate(removed_objs):
            for j, added_obj in enumerate(added_objs):
                tform, icp_score = compute_tform(
                    np.asarray([lp.point for lp in prev_obj.labeled_points]), added_obj
                )
                prev_to_cur_tform[i, j] = tform
                prev_to_cur_dist[i, j] = icp_score

        matching_row, matching_col = linear_sum_assignment(prev_to_cur_dist)
        unmatched_cur = [i for i in range(len(added_objs))]
        matching_list = [[] for _ in range(len(removed_objs))]
        for i, prev_id in enumerate(matching_row):
            matched_id = matching_col[i]
            if prev_to_cur_dist[prev_id, matched_id] > asso_thres:
                continue
            matching_list[prev_id].append(matched_id)
            unmatched_cur.remove(matched_id)
        for i in range(len(added_objs)):
            if len(removed_objs) == 0:
                break
            if i not in unmatched_cur:
                continue
            min_prev_id = prev_to_cur_dist[:, i].argmin()
            min_dist = prev_to_cur_dist[min_prev_id, i]
            if min_dist > asso_thres:
                continue
            matching_list[min_prev_id].append(i)
            unmatched_cur.remove(i)
        for i, matched in enumerate(matching_list):
            if prev_to_cur_dist.shape[1] == 0:
                break
            if len(matched) > 0:
                continue
            min_cur_id = prev_to_cur_dist[i].argmin()
            min_dist = prev_to_cur_dist[i, min_cur_id]
            if min_dist > asso_thres:
                continue
            matching_list[i].append(min_cur_id)

        # add unmatched obj
        for cur_obj_id in unmatched_cur:
            im_y, im_x = np.where(seg_obs[..., 1] == added_objs_segname[cur_obj_id])
            llist = get_llist(im_y, im_x)
            relevant_cloud = [
                lp
                for lp in custom_iterate_point_cloud(
                    camera_image_obs, llist, max_depth=max_depth
                )
            ]
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
            for cluster in clusters:
                cluster = downsample_cluster(cluster)
                if len(cluster) < min_points:
                    continue
                category, bid = cluster[0].label
                min_z = min(lp.point[2] for lp in cluster)
                surface_pose = Pose(Point(z=min_z))
                body = estimate_surface_mesh(
                    cluster,
                    surface_pose,
                    camera_image=camera_image_obs,
                    client=self.client,
                    **kwargs
                )
                if body is None:
                    continue
                self.estimated_objects.append(
                    EstimatedObject(
                        body,
                        category=category,
                        labeled_points=cluster,
                        client=self.client,
                    )
                )

        for i, matched in enumerate(matching_list):
            if len(matched) == 0:
                self.estimated_objects.remove(removed_objs[i])
                removed_objs[i].remove()
                continue
            # asso_cloud = [*(added_objs[matchid]) for matchid in matched]
            asso_cloud = np.vstack([added_objs[matchid] for matchid in matched])
            if len(matched) == 1:
                tform = prev_to_cur_tform[i, matched[0]]
            else:
                tform, dist = compute_tform(
                    np.asarray([lp.point for lp in prev_obj.labeled_points]), asso_cloud
                )
                # TODO asso.
                if dist > asso_thres:
                    import pdb

                    pdb.set_trace()
            update_success = removed_objs[i].update(
                asso_cloud,
                tform,
                camera_image_obs,
                surface=surface,
                min_points=30,
                **kwargs
            )
            if not update_success:
                self.estimated_objects.remove(removed_objs[i])

        cam_sim = self.robot.cameras[0].get_image()
        rgb_sim, _, seg_sim, _, _ = cam_sim
        # import cv2;
        # # import pdb;pdb.set_trace()
        # cv2.imwrite(f'temp_meshes/rgbpred_{len(self.observations)-1}.png',rgb_sim[...,:3][...,::-1])
        # # cv2.imwrite(f'temp_meshes/segpred_{len(self.observations)-1}.png',seg_sim)
        # clusters = [body.labeled_points for body in self.estimated_objects]
        # save_image_from_cluster(clusters, camera_image_obs, min_points=min_points, imname=f'relabeled_proj_{len(self.observations)-1}.png')
        return self.estimated_objects
        # TODO: integrate with surface belief
        # for surface_belief in self.surface_beliefs:
        #     surface_belief.update(camera_image, **kwargs)

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
            surface_pose = Pose(Point(z=min_z))  # if surface is None else surface.pose
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
            self.estimated_objects[i].color = apply_alpha(
                COLOR_FROM_NAME[class_to_name[relabeled_classes[i]]]
            )
        return self.estimated_objects

    def disable(self):
        if self.saver is None:
            self.saver = WorldSaver(bodies=self.estimated_objects)
            for body in self.estimated_objects:
                vector = self.displacement * get_unit_vector([1, 0, 0])
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
