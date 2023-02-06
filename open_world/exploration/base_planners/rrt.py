from open_world.exploration.base_planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (LockRenderer, wait_if_gui, joint_from_name, 
                                                    set_joint_positions, get_aabb)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.patches import Rectangle
import random


class RRT(Planner):
    def __init__(self, env, client=None):
        super(RRT, self).__init__()
        
        self.client=client
        self.env = env

        self.step_size = [0.05, np.pi/18]
        self.RRT_ITERS = 5000

        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]

        self.movable_handles = []

    def get_path(self, start, goal, vis=False, ignore_movable=False, forced_obj_coll=[],
                 attached_object=None, moving_backwards=False):
        with LockRenderer():
            graph = self.rrt(start, goal, n_iter=self.RRT_ITERS,
                             ignore_movable=ignore_movable,
                             forced_object_coll=forced_obj_coll,
                             attached_object=attached_object,
                             moving_backwards=moving_backwards)
            #if moving_backwards:
            plot(graph, self.env)
            if(not graph.success):
                return None


            path = dijkstra(graph)
            if not moving_backwards:
                final_path = self.env.adjust_angles(path, start, goal)
            else:
                final_path = self.env.adjust_angles(path, goal, start)
                final_path.reverse()
        if vis:
            plot(graph, path=final_path)

        return final_path

    def get_plan(self, **kwargs):
        
        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data)
        self.env.update_occupancy(image_data)


        self.movable_handles = self.env.plot_grids(visibility=False, occupancy=True, movable=True)


        current_q, complete = self.env.start, False

        while(not complete):
            final_path = self.get_path(current_q, self.env.goal)
            if(final_path is None):
                print("No path to goal :(")
                break
            current_q, complete = self.execute_path(final_path)

        wait_if_gui()

    def execute_path(self, path, ignore_movable=False):
        for qi, q in enumerate(path):
            set_joint_positions(self.env.robot, self.joints, q)

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(image_data)
            self.env.update_movable_boxes(image_data)
            self.env.update_visibility(camera_pose, image_data)

            # Check if remaining path is collision free under the new occupancy grid
            for next_qi in path[qi:]:
                if(self.env.check_conf_collision(next_qi, ignore_movable=ignore_movable)):
                    self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                    return q, False
        return q, True


    def rrt(self, start_node, goal_node, n_iter=500, radius = 0.3, goal_bias=0.1,
            ignore_movable=False, forced_object_coll=[], attached_object=None, moving_backwards=False):

        start, goal = start_node, goal_node
        if moving_backwards:
            goal, start = start, goal

        lower, upper = self.env.room.aabb
        G = Graph(start, goal)

        for _ in range(n_iter):
            goal_sample = np.random.choice([True, False], 1, p=[goal_bias, 1-goal_bias])[0]
            if goal_sample:
                rand_vex = goal
            else:
                rand_vex = self.sample(lower, upper)
            new_vex = None

            for k in range(2):
                near_vex, near_idx = G.nearest_vex(rand_vex, self.env, self.joints,k=k)

                if near_vex is None:
                    break

                new_vex = self.steer(near_vex, rand_vex, moving_backwards=False)

                if self.env.check_collision_in_path(near_vex, new_vex,
                                                    ignore_movable=ignore_movable,
                                                    forced_object_coll=forced_object_coll,
                                                    attached_object=attached_object,
                                                    moving_backwards=False):
                    new_vex = None
                else:
                    break

            if new_vex is None or near_vex is None:
                continue

            new_idx = G.add_vex(new_vex)
            dist = distance(new_vex, near_vex)
            G.add_edge(new_idx, near_idx, dist)

            dist_to_goal = distance(new_vex, G.endpos)

            if dist_to_goal < radius:
                end_idx = G.add_vex(G.endpos)
                G.add_edge(new_idx, end_idx, dist_to_goal)
                G.success = True
                return G

        return G


    def sample(self, lower_limit, upper_limit):
        rand_x = np.random.uniform(lower_limit[0], upper_limit[0])
        rand_y = np.random.uniform(lower_limit[1], upper_limit[1])
        rand_t = np.random.uniform(0, 2*np.pi)
        return (rand_x, rand_y, rand_t)


    def sample_from_vision(self):
        resolution = self.env.visibility_grid.resolutions
        free_points = [(free[0]*resolution[0], free[1]*resolution[1])
                       for free in self.env.viewed_voxels]
        point = random.choice(free_points)
        rand_t = np.random.uniform(0, 2 * np.pi)

        return (point[0], point[1], rand_t)


    def steer(self, source_vex, dest_vex, step_size=0.1, moving_backwards=False):
        dirn = np.array(dest_vex[0:2]) - np.array(source_vex[0:2])
        length = np.linalg.norm(dirn)
        dirn = (dirn / length) * min(step_size, length)

        beg, end = source_vex, dest_vex
        if moving_backwards:
            beg, end = dest_vex, source_vex

        delta_x = end[0] - beg[0]
        delta_y = end[1] - beg[1]
        theta = np.arctan2(delta_y, delta_x)

        new_vex = (source_vex[0]+dirn[0], source_vex[1]+dirn[1], theta)
        return new_vex


class Graph:
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos: 0}
        self.neighbors = {0: []}
        self.distances = {0: 0.0}


    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx


    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))


    def nearest_vex(self, vex, environment, joints, k=0):
        neighbors = [(distance(v,vex), v, idx) for idx,v in enumerate(self.vertices)]
        neighbors = sorted(neighbors, key=lambda x: x[0])

        if len(neighbors) < k+1:
            return None, None

        return neighbors[k][1], neighbors[k][2]


def distance(vex1, vex2):
    return ((vex1[0] - vex2[0])**2 + (vex1[1]-vex2[1])**2)**0.5


def plot(G, env, path=None):
    '''
    Plot RRT, obstacles and shortest path
    '''
    px = [x for x, y, t in G.vertices]
    py = [y for x, y, t in G.vertices]
    pt = [t for x, y, t in G.vertices]
    fig, ax = plt.subplots()

    ax.scatter(px, py, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], c='red')

    lines = [(G.vertices[edge[0]][0:2], G.vertices[edge[1]][0:2]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    # Draw angles of points
    angle_lines = []
    for x,y,t in G.vertices:
        endy = y + 0.05 * np.sin(t)
        endx = x+ 0.05 * np.cos(t)
        angle_lines.append(((x,y), (endx, endy)))
    lc = mc.LineCollection(angle_lines, colors='red', linewidths=2)
    ax.add_collection(lc)

    # Draw room shape
    for wall in env.room.walls:
        wall_aabb = get_aabb(wall)
        rec = Rectangle((wall_aabb.lower[0:2]),
                     wall_aabb.upper[0] - wall_aabb.lower[0],
                     wall_aabb.upper[1] - wall_aabb.lower[1],
                     color="grey", linewidth=0.1)
        ax.add_patch(rec)

    # Not taking rotations into account
    for obstacle in env.static_objects + env.movable_boxes:
        color = "brown"
        if isinstance(obstacle, int):
            aabb = get_aabb(obstacle)
        else:
            aabb = obstacle.aabb
            color = "yellow"
        ax.add_patch(Rectangle((aabb.lower[0], aabb.lower[1]),
                     aabb.upper[0] - aabb.lower[0],
                     aabb.upper[1] - aabb.lower[1],
                     color=color, linewidth=0.1))


    if path is not None:
        paths = [(path[i][0:2], path[i+1][0:2]) for i in range(len(path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()

def dijkstra(G):
    '''
        Dijkstra algorithm for finding shortest path from start position to end.
    '''
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)
