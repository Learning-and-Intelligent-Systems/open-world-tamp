import time
from itertools import groupby

import numpy as np
from open_world.exploration.base_planners.planner import Planner
from open_world.exploration.utils import GRID_RESOLUTION, find_min_angle
from open_world.exploration.utils_graph import Graph
from pybullet_planning.pybullet_tools.utils import joint_from_name


class AStarSearch(Planner):
    def __init__(self, env, client=None):
        self.env = env

        self.client = client
        # Initializes a graph that contains the available movements
        self.G = Graph()
        self.G.initialize_full_graph(
            self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi / 8]
        )

        # In case there is an environment with restricted configurations
        self.env.restrict_configuration(self.G)

        # Creates a voxel structure that contains the vision space
        self.env.setup_default_vision(self.G)

        # Specific joints to move the robot in simulation
        self.joints = [
            joint_from_name(self.env.robot, "x", client=self.client),
            joint_from_name(self.env.robot, "y", client=self.client),
            joint_from_name(self.env.robot, "theta", client=self.client),
        ]

        # Structure used to save voxels that cannot be accessed by the robot, hence occupied
        self.occupied_voxels = dict()
        self.debug = False
        self.v_0 = None

    def get_plan(self, debug=False, **kwargs):
        self.debug = debug
        q_start, q_goal = self.env.start, self.env.goal
        # Gets initial vision and updates the current vision based on it
        self.v_0 = self.env.get_circular_vision(q_start, self.G)
        self.env.update_vision_from_voxels(self.v_0)

        # Gathers vision from the robot's starting position and updates the
        # visibility and occupancy grids. Visualize them for convenience.
        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data, q_start)
        self.env.update_occupancy(q_start, image_data)
        self.env.update_movable_boxes(image_data)
        self.env.plot_grids(True, True, True)

        complete = False
        current_q = q_start
        final_executed = []
        while not complete:
            path = self.a_star(current_q, q_goal)

            if path is None:
                print("Can't find a path to goal")
                return [key for key, _group in groupby(final_executed)]
            current_q, complete, _, executed_path = self.execute_path(path)
            final_executed += executed_path

        # Search for repeated nodes in a sequence and filter them.
        return [key for key, _group in groupby(final_executed)]

    def action_fn(self, q, extended=set()):
        """Helper function to the search, that given a node, it gives all the
        possible actions to take with the inquired cost of each. Uses the
        vision constraint on each node based on the vision gained from the
        first path found to the node.

        Args:
            q (tuple): The node to expand.
            extended (set): Set of nodes that were already extended by the search.
        Returns:
            list: A list of available actions with the respective costs.
        """
        actions = []
        # Retrieve all the neighbors of the current node based on the graph of the space.
        for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
            q_prime = self.G.vertices[q_prime_i]
            # If the node has already been extended do not consider it.
            if q_prime in extended:
                continue

            # Restrict A* to only move forward or rotate
            angle = np.arctan2(q_prime[1] - q[1], q_prime[0] - q[0])
            angle = round(angle + 2 * np.pi, 3) if angle < 0 else round(angle, 3)
            if q[:2] != q_prime[:2] and (angle != q[2]):
                continue

            # Check for whether the new node is in obstruction with any obstacle.
            collisions, coll_objects = self.env.obstruction_from_path(
                [q, q_prime], set()
            )
            if not collisions.shape[0] > 0 and coll_objects is None:
                actions.append((q_prime, distance(q, q_prime)))
        return actions

    def a_star(self, q_start, q_goal):
        """A* search algorithm.

        Args:
            q_start (tuple): Start node.
            q_goal (tuple): Goal node.
        Returns:
            list: The path from start to goal.
        """
        # Timing the search for benchmarking purposes.
        current_t = time.time()
        extended = set()
        paths = [([q_start], 0, 0)]

        while paths:
            current = paths.pop(-1)
            best_path = current[0]
            best_path_cost = current[1]

            # Ignore a node that has already been extended.
            if best_path[-1] in extended:
                continue

            # If goal is found return it, graph the search, and output the elapsed time.
            if (
                np.linalg.norm(np.array(list(best_path[-1])) - np.array(list(q_goal)))
                < 0.01
            ):
                done = time.time() - current_t
                print("Extended nodes: {}".format(len(extended)))
                print("Search Time: {}".format(done))
                if self.debug:
                    self.G.plot_search(self.env, extended, path=best_path, goal=q_goal)
                return best_path

            extended.add(best_path[-1])
            actions = self.action_fn(best_path[-1], extended=extended)
            for action in actions:
                paths.append(
                    (
                        best_path + [action[0]],
                        best_path_cost + action[1],
                        distance(action[0], q_goal),
                    )
                )

            # Only sorting from heuristic. Faster but change if needed
            paths = sorted(paths, key=lambda x: x[-1] + x[-2], reverse=True)

        done = time.time() - current_t
        print("Extended nodes: {}".format(len(extended)))
        print("Search Time: {}".format(done))
        if self.debug:
            self.G.plot_search(self.env, extended, goal=q_goal)
        return None

    def execute_path(self, path):
        """Executes a given path in simulation until it is complete or no
        longer feasible.

        Args:
            path (list): The path to execute.
        Returns:
            tuple: A tuple containing the state where execution stopped, whether it was able to reach the goal,
             the gained vision, and the executed path.
        """
        gained_vision = set()
        executed = []
        for qi, q in enumerate(path):
            self.env.move_robot(q, self.joints)
            # Executed paths saved as a list of q and attachment.
            executed.append([q, None])

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(q, image_data)
            gained_vision.update(self.env.update_movable_boxes(image_data))
            gained_vision.update(self.env.update_visibility(camera_pose, image_data, q))

            # Check if remaining path is collision free under the new occupancy grid
            obstructions, collided_obj = self.env.obstruction_from_path(
                path[qi:], set()
            )
            if obstructions.shape[0] > 0 or collided_obj is not None:
                print("Found a collision on this path. Aborting")
                self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                return q, False, gained_vision, executed
            self.env.plot_grids(visibility=True, occupancy=True, movable=True)

        return q, True, gained_vision, executed

    def adjust_angles(self, path):
        final_path = [path[0]]
        for i in range(1, len(path)):
            if path[i - 1][:2] == path[i][:2]:
                continue
            curr_angle = final_path[-1][2]
            angle = np.arctan2(path[i][1] - path[i - 1][1], path[i][0] - path[i - 1][0])
            angle = angle + 2 * np.pi if angle < 0 else angle
            angle_traverse = 1 if find_min_angle(curr_angle, angle) > 0 else -1
            while round(curr_angle, 3) != round(angle, 3):
                curr_node = (final_path[-1][0], final_path[-1][1], curr_angle)
                curr_angle = self.G.get_vertex_rot(curr_node, angle_traverse)[2]
                final_path.append(
                    (path[i - 1][0], path[i - 1][1], round(curr_angle, 3))
                )
            final_path.append((path[i][0], path[i][1], round(curr_angle, 3)))

        if final_path[-1] != path[-1]:
            curr_angle = final_path[-1][2]
            angle = path[-1][2]
            angle_traverse = 1 if find_min_angle(curr_angle, angle) > 0 else -1
            while round(curr_angle, 3) != round(angle, 3):
                curr_node = (final_path[-1][0], final_path[-1][1], curr_angle)
                curr_angle = self.G.get_vertex_rot(curr_node, angle_traverse)[2]
                final_path.append(
                    (final_path[-1][0], final_path[-1][1], round(curr_angle, 3))
                )

        return final_path


def distance(vex1, vex2):
    """Helper function that returns the Euclidean distance between two
    configurations. It uses a "fudge" factor for the relationship between
    angles and distances.

    Args:
        vex1 (tuple): The first tuple
        vex2 (tuple): The second tuple
    Returns:
        float: The Euclidean distance between both tuples.
    """
    r = 0.01
    dist = 0
    for i in range(len(vex1) - 1):
        dist += (vex1[i] - vex2[i]) ** 2
    dist += (r * find_min_angle(vex1[2], vex2[2])) ** 2
    return dist**0.5
