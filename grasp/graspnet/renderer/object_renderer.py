from __future__ import print_function
from __future__ import absolute_import

import trimesh
import trimesh.transformations as tra
import pyrender
import numpy as np
import copy
import cv2
import h5py
import utils.sample as sample
import math
import sys
import argparse
import os


class ObjectRenderer:
    def __init__(self, fov=np.pi / 6, object_paths=[], object_scales=[]):
        """
        Args:
          fov: float, 
        """
        self._fov = fov
        self.mesh = None
        self._scene = None
        self.tmesh = None
        self._init_scene()
        self._object_nodes = []
        self._object_means = []
        self._object_distances = []

        assert (isinstance(object_paths, list))
        assert (len(object_paths) > 0)

        for path, scale in zip(object_paths, object_scales):
            node, obj_mean = self._load_object(path, scale)
            self._object_nodes.append(node)
            self._object_means.append(obj_mean)

    def _init_scene(self):
        self._scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(
            yfov=self._fov, aspectRatio=1.0,
            znear=0.001)  # do not change aspect ratio
        camera_pose = tra.euler_matrix(np.pi, 0, 0)

        self._scene.add(camera, pose=camera_pose, name='camera')

        light = pyrender.SpotLight(color=np.ones(4),
                                   intensity=3.,
                                   innerConeAngle=np.pi / 16,
                                   outerConeAngle=np.pi / 6.0)
        self._scene.add(light, pose=camera_pose, name='light')

        self.renderer = pyrender.OffscreenRenderer(400, 400)

    def _load_object(self, path, scale=1.0):
        obj = sample.Object(path)
        obj.rescale(scale)
        print('rescaling with scale', scale)

        tmesh = obj.mesh
        tmesh_mean = np.mean(tmesh.vertices, 0)
        tmesh.vertices -= np.expand_dims(tmesh_mean, 0)

        lbs = np.min(tmesh.vertices, 0)
        ubs = np.max(tmesh.vertices, 0)
        self._object_distances.append(np.max(ubs - lbs) * 5)

        print(self._object_distances)

        self.tmesh = copy.deepcopy(tmesh)
        mesh = pyrender.Mesh.from_trimesh(tmesh)

        return self._scene.add(mesh,
                               name='object'), np.expand_dims(tmesh_mean, 0)

    def _to_pointcloud(self, depth):
        fy = fx = 0.5 / np.tan(self._fov * 0.5)  # aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)

        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T

    def render(self, object_poses, render_pc=True):
        assert (isinstance(object_poses, list))
        assert (len(object_poses) == len(self._object_nodes))

        all_transferred_poses = []
        for object_pose, object_node, object_distance in zip(
                object_poses, self._object_nodes, self._object_distances):
            transferred_pose = object_pose.copy()
            transferred_pose[2, 3] = object_distance
            all_transferred_poses.append(transferred_pose)
            self._scene.set_pose(object_node, transferred_pose)

        color, depth = self.renderer.render(self._scene)

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None

        return color, depth, pc, all_transferred_poses

    def render_all_and_save_to_h5(self, output_path, all_eulers, vis=False):
        """
        Args:
           output_path: path of the h5 file.
           all_eulers: list of 3 elemenet-tuples indicating the euler angles
             in degrees.
        """
        if len(self._object_nodes) != 1:
            raise ValueError(
                'object nodes should have 1 element, not {}'.format(
                    len(self._object_nodes)))

        hf = h5py.File(output_path)
        #point_grp = hf.create_group('points')
        #pose_grp = hf.create_group('object_poses')
        mean_grp = hf.create_dataset('object_mean', data=self._object_means[0])

        pcs = []
        rotations = []
        #import mayavi.mlab as mlab
        for i, euler in enumerate(all_eulers):
            assert isinstance(euler, tuple) and len(euler) == 3
            rotation = tra.euler_matrix(*euler)
            color, _, pc, final_rotation = self.render([rotation])
            MAX_POINTS = 3000
            if pc.shape[0] > MAX_POINTS:
                pc = pc[np.random.choice(
                    range(pc.shape[0]), replace=False, size=MAX_POINTS), :]
            elif pc.shape[0] < MAX_POINTS:
                pc = pc[np.random.choice(
                    range(pc.shape[0]), replace=True, size=MAX_POINTS), :]

            #print('{}/{}: {}'.format(i, len(all_eulers), pc.shape))
            cv2.imshow('w', color)
            cv2.waitKey(1)

            # mlab.figure()
            #mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2])
            # mlab.show()
            key = '{}_{}_{}'.format(euler[0], euler[1], euler[2])
            pcs.append(pc)
            rotations.append(final_rotation[0])

        hf.create_dataset('pcs', data=pcs, compression='gzip')
        hf.create_dataset('object_poses', data=rotations, compression='gzip')

        hf.close()

    @property
    def object_distances(self):
        return self._object_distances
