from open_world.exploration.base_planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import wait_if_gui
from open_world.exploration.utils_graph import Graph
import numpy as np

GRID_RESOLUTION = 0.2

class DoNothing(Planner):
    def __init__(self, env):
        super(DoNothing, self).__init__()

        self.env = env

    def get_plan(self, **kwargs):

        G = Graph()
        G.initialize_full_graph(self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi/8])
        self.env.setup_default_vision(G)
        self.env.restrict_configuration(G)
        # for wall in self.env.room.walls:
        #     for voxel in self.env.occupancy_grid.voxels_from_aabb(scale_aabb(get_aabb(wall), 0.98)):
        #         self.env.occupancy_grid.set_occupied(voxel)

        camera_pose, image_data = self.env.get_robot_vision()
        # self.env.update_visibility(camera_pose, image_data, self.env.start)
        voxels = self.env.get_optimistic_vision(self.env.start, G)
        for voxel in voxels:
            self.env.visibility_grid.set_free(voxel)
        self.env.update_occupancy(self.env.start, image_data)
        self.env.update_movable_boxes(image_data)
        # self.env.plot_grids(True, True, True)

        G.plot(self.env)
        wait_if_gui()
