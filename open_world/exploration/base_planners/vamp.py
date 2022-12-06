from itertools import groupby

from open_world.exploration.base_planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (set_joint_positions,
                                                    joint_from_name, get_aabb_volume, set_pose, get_pose)
import numpy as np
import time
import datetime
import scipy.spatial
import pickle
from utils_graph import Graph
from open_world.exploration.utils import GRID_RESOLUTION, find_min_angle

class Vamp(Planner):
    def __init__(self, env):

        # Sets up the environment and necessary data structures for the planner
        super(Vamp, self).__init__()

        self.env = env

        # Initializes a graph that contains the available movements
        self.G = Graph()
        self.G.initialize_full_graph(self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi/8])

        # In case there is an environment with restricted configurations
        self.env.restrict_configuration(self.G)

        # Creates a voxel structure that contains the vision space
        self.env.setup_default_vision(self.G)

        # Specific joints to move the robot in simulation
        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]

        # Structure used to save voxels that cannot be accessed by the robot, hence occupied
        self.occupied_voxels = dict()
        self.debug = False
        self.v_0 = None
        self.R = None
        self.final_executed = []
        self.object_poses = None
        self.current_q = None
        self.vision_q = dict()

    def get_plan(self, loadfile=None, debug=False, **kwargs):
        """
        Creates a plan and executes it based on the given planner and environment.

        Args:
            loadfile (str): Location of the save file to load containing a previous state.
        Returns:
            list: The plan followed by the robot from start to goal.
        """
        self.debug = debug
        self.current_q, q_goal = self.env.start, self.env.goal
        # Gets initial vision and updates the current vision based on it
        self.v_0 = self.env.get_circular_vision(self.current_q, self.G)
        self.env.update_vision_from_voxels(self.v_0)

        # Gathers vision from the robot's starting position and updates the
        # visibility and occupancy grids. Visualize them for convenience.
        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data, self.current_q)
        self.env.update_occupancy(self.current_q, image_data)
        self.env.update_movable_boxes(image_data)
        self.env.plot_grids(True, True, True)



        # In case a loadfile is given. Load the state of the program to the specified one.
        if loadfile is not None:
            self.load_state(loadfile)
            self.env.plot_grids(True, True, True)
            set_joint_positions(self.env.robot, self.joints, self.current_q)
            for i, obj in enumerate(self.env.room.movable_obstacles):
                set_pose(obj, self.object_poses[i])
            print("State loaded")


        complete = False
        # Continue looking for a plan until the robot has reached the goal.
        while not complete:
            path = self.vamp_backchain(self.current_q, q_goal, self.v_0)
            # If at any point there is no possible path, then the search is ended.
            if path is None:
                return [key for key, _group in groupby(self.final_executed)]

            # Execute path until it fails due to obstruction, or it reaches the goal. Update the
            # visibility based on what was observed while traversing the path.
            self.current_q, complete, gained_vision, executed_path = self.execute_path(path)
            self.final_executed += executed_path
            self.v_0.update(gained_vision)

            if self.debug:
                print("Want to save this state? Press Y or N then Enter")
                x = input()
                if x == "Y" or x == "y":
                    self.object_poses = []
                    for obj in self.env.room.movable_obstacles:
                        self.object_poses.append(get_pose(obj))
                    self.save_state()

        # Search for repeated nodes in a sequence and filter them.
        return [key for key, _group in groupby(self.final_executed)]

    def vamp_backchain(self, q_start, q_goal, v_0):
        """
        Main function for path planning using VAMP.

        Args:
            q_start (tuple): Initial position from where to start to plan.
            q_goal (tuple): Goal position where the planning is ended.
            v_0 (set): Set of voxels which indicate what areas of the world have been seen before
                    initializing the planning
        Returns:
            list: A suggested plan for the robot to follow.
        """
        p = []
        v = v_0
        q = q_start

        while True:
            # Find a path to goal, keeping the visualization constraint and return it if found
            p_final = self.vamp_path_vis(q, q_goal, v)
            if p_final is not None:
                return p + p_final
            print("Couldn't find a direct path. Looking for a relaxed one")
            # If a path to goal can't be found, find a relaxed path and use it as a subgoal
            p_relaxed = self.vamp_path_vis(q, q_goal, v, relaxed=True)
            if p_relaxed is None:
                return None
            print("Looking for path to subgoal")
            # Continue until either we see all the required area or the relaxed path does not
            # exist. This uses the assumption that we can see every voxel from some configuration
            required_viewing = self.env.visibility_voxels_from_path(p_relaxed).difference(v)
            while len(required_viewing) > 0:
                p_vis = self.vavp(q, required_viewing, v)
                # If the relaxed version fails, explore some of the environment. And restart the search
                if p_vis is None:
                    print("P_VIS failed. Observing some of the environment")
                    W = set(self.env.static_vis_grid.value_from_voxel.keys())
                    p_vis = self.tourist(q, W.difference(v), v)
                if p_vis is None:
                    print("P_VIS failed again. Aborting")
                    return None
                p += p_vis
                v = v.union(self.env.get_optimistic_path_vision(p_vis, self.G))
                q = p_vis[-1]

                # Check if we made progress seen the area, if we did, then don't change the p_relaxed
                # Otherwise, find a new p_relaxed
                required_viewing_check = self.env.visibility_voxels_from_path(p_relaxed).difference(v)
                if len(required_viewing) != len(required_viewing_check):
                    required_viewing = required_viewing_check
                else:
                    p_relaxed = self.vamp_path_vis(q, q_goal, v, relaxed=True)
                    if p_relaxed is None:
                        break
                    required_viewing = self.env.visibility_voxels_from_path(p_relaxed).difference(v)


    def vavp(self, q, R, v, obstructions=set()):
        """
        Subprocedure to aid the planning on dividing the objective into subgoals and planning paths
        accordingly.

        Args:
            q (tuple): Current position of the robot.
            R (set) : Set of voxels that define the area we are interested in gaining vision from.
            v (set) : Set of tuples that define the already seen space.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: The suggested path that views some of the area of interest.
        """
        # Try visualizing the area of interest keeping the vision constraint.
        p_vis = self.tourist(q, R, v)
        if p_vis is not None:
            return p_vis
        # If can't view the area, find a relaxed path that does the same and make this new path
        # the new subgoal. Call the function recursively.
        obstructions_new = obstructions.union(R)
        p_relaxed = self.tourist(q, R, v, relaxed=True, obstructions=obstructions_new)
        if p_relaxed is not None:
            p_vis = self.vavp(q, self.env.visibility_voxels_from_path(p_relaxed).difference(v), v, obstructions=obstructions_new)
            if p_vis is not None:
                return p_vis
        return None

    def sample_goal_from_required(self, R, obstructions=set()):
        """
        Samples a goal position that views most of the required region

        Args:
            R (set) : Set of voxels that define the area we are interested in gaining vision from.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
            ignore_movable (bool): Whether to ignore collisions with movable objects or not.
        Returns:
            A configuration where the robot sees most of the space, from a set of random configurations.
        """
        q_goal = None
        score = 0
        number_of_samples = 1000
        # Sample a goal position that views most of the space of interest.
        rand_qs = self.G.rand_vex(self.env, samples=number_of_samples)
        for q_rand in rand_qs:
            # Check collisions with obstacle and movable objects if required
            collisions, coll_objects = self.env.obstruction_from_path([q_rand], obstructions)
            if not collisions.shape[0] > 0 and coll_objects is None:
                new_score = len(self.env.get_optimistic_vision(q_rand, self.G, obstructions=obstructions).intersection(R))
                if new_score != 0:
                    if new_score > score:
                        q_goal = q_rand
                        score = new_score
        return q_goal

    def tourist(self, q_start, R, v_0, q_goal=None, relaxed=False, obstructions=set()):
        """
        Procedure used to find a path that partially or completely views some area of interest.

        Args:
            q_start (tuple): Starting position of the robot.
            R (set) : Set of voxels that define the area we are interested in gaining vision from.
            v_0 (set) : Set of tuples that define the already seen space.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: The suggested path that views some of the area of interest.
        """
        # Defines a heuristic function to use on the A* star search. Currently using the distance to goal.
        def heuristic_fn(q):
            # Previously used code that defines the heuristic as the smallest distance from the vision
            # gained in the configuration to the area of interest.
            vision_q = self.env.get_optimistic_vision(q, self.G)
            if len(R.intersection(vision_q)) != 0:
                return 0
            if len(vision_q) == 0:
                return (self.env.room.aabb.upper[0] - self.env.room.aabb.lower[0])/2
            s1 = np.array(list(vision_q))
            s2 = np.array(list(R))
            return scipy.spatial.distance.cdist(s1, s2).min()*GRID_RESOLUTION

        return self.vamp_path_vis(q_start, q_goal, v_0, H=heuristic_fn, relaxed=relaxed, obstructions=obstructions,
                                  from_required=True)


    def vamp_step_vis(self, q_start, q_goal, v_0, H=0, relaxed=False, obstructions=set()):
        """
        Helper function to initialize the search. Uses the vision constraint on each node based
        on the vision gained from the previous step only.

        Args:
            q_start (tuple): Starting position of the robot.
            q_goal (tuple): Goal position where the planning is ended.
            v_0 (set) : Set of tuples that define the already seen space.
            H (function): Function defining the heuristic used during A* search.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: The suggested path that goes from start to goal.
        """
        if H == 0:
            H = lambda x: distance(x, q_goal)

        return self.a_star(q_start, q_goal, v_0, H, relaxed, self.action_fn_step, obstructions=obstructions)


    def vamp_path_vis(self, q_start, q_goal, v_0, H=0, relaxed=False, obstructions=set(), from_required=False):
        """
        Helper function to initialize the search. Uses the vision constraint on each node based
        on the vision gained from the first path found to the node.

        Args:
            q_start (tuple): Starting position of the robot.
            q_goal (tuple): Goal position where the planning is ended.
            v_0 (set): Set of tuples that define the already seen space.
            H (function): Function defining the heuristic used during A* search.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: The suggested path that goes from start to goal.
        """
        if H == 0:
            H = lambda x: distance(x, q_goal)
        self.vision_q = dict()

        return self.a_star(q_start, q_goal, v_0, H, relaxed, self.action_fn_path, obstructions=obstructions,
                           from_required=from_required)

    def action_fn_step(self, path, v_0, relaxed=False, extended=set(), obstructions=set()):
        """
        Helper function to the search, that given a node, it gives all the possible actions to take with
        the inquired cost of each. Uses the vision constraint on each node based
        on the vision gained from the previous step only.

        Args:
            path (list): The path obtained to reach the current node on the search.
            v_0 (set): Set of tuples that define the already seen space.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            extended (set): Set of nodes that were already extended by the search.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: A list of available actions with the respective costs.
        """
        actions = []
        q = path[-1]
        # Retrieve all the neighbors of the current node based on the graph of the space.
        for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
            q_prime = self.G.vertices[q_prime_i]
            # If the node has already been extended do not consider it.
            if q_prime in extended:
                continue
            if relaxed:
                # Check for whether the new node is in obstruction with any obstacle.
                if not self.env.obstruction_from_path([q, q_prime], obstructions):
                    v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    s_q = self.env.visibility_voxels_from_path([q, q_prime])
                    # If the node follows the visibility constraint, add it normally.
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
                    # If it does not follow the visibility constraint, add it with a special cost.
                    else:
                        cost = distance(q, q_prime) *\
                                abs(self.volume_from_voxels(self.env.static_vis_grid, s_q.difference(v_q)))
                        actions.append((q_prime, cost))
            else:
                # In the not relaxed case only add nodes when the visibility constraint holds.
                if not self.env.obstruction_from_path([q, q_prime], obstructions):
                    v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    s_q = self.env.visibility_voxels_from_path([q, q_prime])
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
        return actions

    def action_fn_path(self, path, v_0, relaxed=False, extended=set(), obstructions=set()):
        """
        Helper function to the search, that given a node, it gives all the possible actions to take with
        the inquired cost of each. Uses the vision constraint on each node based
        on the vision gained from the first path found to the node.

        Args:
            path (list): The path obtained to reach the current node on the search.
            v_0 (set): Set of tuples that define the already seen space.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            extended (set): Set of nodes that were already extended by the search.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: A list of available actions with the respective costs.
        """
        actions = []
        q = path[-1]
        # Retrieve all the neighbors of the current node based on the graph of the space.
        for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
            q_prime = self.G.vertices[q_prime_i]
            # If the node has already been extended do not consider it.
            if q_prime in extended:
                continue
            if relaxed:
                # Check for whether the new node is in obstruction with any obstacle.
                collisions, coll_objects = self.env.obstruction_from_path([q, q_prime], obstructions)

                if not collisions.shape[0] > 0 and coll_objects is None:
                    if len(path) == 1:
                        v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    else:
                        v_q = self.vision_q[path[-2]].union(self.env.get_optimistic_vision(q, self.G))
                    self.vision_q[q] = v_q
                    s_q = self.env.visibility_voxels_from_path([q, q_prime])
                    # If the node follows the visibility constraint, add it normally.
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
                    # If it does not follow the visibility constraint, add it with a special cost.
                    else:
                        #cost = distance(q, q_prime) *\
                        #        abs(self.volume_from_voxels(self.env.static_vis_grid, s_q.difference(v_q)))*1000
                        cost = distance(q, q_prime) * len(s_q.difference(v_q))
                        actions.append((q_prime, cost))
            else:
                # In the not relaxed case only add nodes when the visibility constraint holds.
                collisions, coll_objects = self.env.obstruction_from_path([q, q_prime], obstructions)

                if not collisions.shape[0] > 0 and coll_objects is None:
                    if len(path) == 1:
                        v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    else:
                        v_q = self.vision_q[path[-2]].union(self.env.get_optimistic_vision(q, self.G))
                    self.vision_q[q] = v_q
                    s_q = self.env.visibility_voxels_from_path([q, q_prime])
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
        return actions


    def volume_from_voxels(self, grid, voxels):
        """
        Calculates the volume of a given set of voxels.

        Args:
            grid (object): The grid to which the voxels belong to.
            voxels (set): The set of voxels from which to determine the volume
        Returns:
            float: The volume of the voxels.
        """
        if len(voxels) == 0:
            return 0
        voxel_vol = get_aabb_volume(grid.aabb_from_voxel(next(iter(voxels))))
        return voxel_vol*len(voxels)



    def a_star(self, q_start, q_goal, v_0, H, relaxed, action_fn, obstructions=set(), from_required=False):
        """
        A* search algorithm.

        Args:
            q_start (tuple): Start node.
            q_goal (tuple): Goal node.
            v_0 (set): Set of tuples that define the already seen space.
            H (function): Function defining the heuristic used during search.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            action_fn (function): Defines the possible actions that any node can take and their costs.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: The path from start to goal.
        """
        try:
            # Timing the search for benchmarking purposes.
            current_t = time.time()
            extended = set()
            paths = [([q_start], 0, 0)]

            if from_required:
                q_goal = None

            while paths:
                current = paths.pop(-1)
                best_path = current[0]
                best_path_cost = current[1]

                # Ignore a node that has already been extended.
                if best_path[-1] in extended:
                    continue

                # If goal is found return it, graph the search, and output the elapsed time.
                if not from_required:
                    if np.linalg.norm(np.array(list(best_path[-1]))-np.array(list(q_goal)))<0.01:
                        done = time.time() - current_t
                        print("Extended nodes: {}".format(len(extended)))
                        print("Search Time: {}".format(done))
                        if self.debug:
                            self.G.plot_search(self.env, extended, path=best_path, goal=q_goal)
                        return best_path
                else:
                    if H(best_path[-1]) == 0:
                        q_goal = best_path[-1]
                        done = time.time() - current_t
                        print("Extended nodes: {}".format(len(extended)))
                        print("Search Time: {}".format(done))
                        if self.debug:
                            self.G.plot_search(self.env, extended, path=best_path, goal=q_goal)
                        return best_path

                extended.add(best_path[-1])
                actions = action_fn(best_path, v_0, relaxed=relaxed, extended=extended, obstructions=obstructions)
                for action in actions:
                    paths.append((best_path + [action[0]], best_path_cost + action[1], H(action[0])))

                # TODO: Only sorting from heuristic. Faster but change if needed
                paths = sorted(paths, key=lambda x: x[-1] + x[-2], reverse=True)

            done = time.time() - current_t
            print("Extended nodes: {}".format(len(extended)))
            print("Search Time: {}".format(done))
            if self.debug:
                self.G.plot_search(self.env, extended, goal=q_goal)
            return None
        except KeyboardInterrupt:
            return None


    def execute_path(self, path):
        """
        Executes a given path in simulation until it is complete or no longer feasible.

        Args:
            path (list): The path to execute.
        Returns:
            tuple: A tuple containing the state where execution stopped, whether it was able to reach the goal,
             the gained vision, and the executed path.
        """
        gained_vision = set()
        executed = []
        for qi, q in enumerate(path):
            # Check whether the next step goes into area that is unseen.
            next_occupied = self.env.visibility_voxels_from_path([q])
            for voxel in next_occupied:
                if self.env.visibility_grid.contains(voxel):
                    qi = qi-1 if qi-1 >= 0 else 0
                    print("Stepping into unseen area. Aborting")
                    return path[qi], False, gained_vision, executed

            self.env.move_robot(q, self.joints)
            # Executed paths saved as a list of q and attachment.
            executed.append([q, None])

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(q, image_data)
            gained_vision.update(self.env.update_movable_boxes(image_data))
            gained_vision.update(self.env.update_visibility(camera_pose, image_data, q))

            # Check if remaining path is collision free under the new occupancy grid
            obstructions, collided_obj = self.env.obstruction_from_path(path[qi:], set())
            if obstructions.shape[0] > 0 or collided_obj is not None:
                print("Found a collision on this path. Aborting")
                self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                return q, False, gained_vision, executed
            self.env.plot_grids(visibility=True, occupancy=True, movable=True)
        return q, True, gained_vision, executed

    def save_state(self):
        """
        Saves the current state of the Vamp planning algorithm.
        """
        current_time = datetime.datetime.now()
        dbfile = open("saves/{}_state_{}_{}_{}_{}_{}_{}.dat".format(self.env.__class__.__name__,
                                                                    self.__class__.__name__, current_time.month,
                                                                    current_time.day, current_time.hour,
                                                                    current_time.minute, current_time.second), "wb")
        pickle.dump(self, dbfile)
        dbfile.close()

    def load_state(self, filename):
        """
        Loads the specified file containing a state of the Vamp planner.

        Args:
            filename (str): The path to the file to load.
        """
        dbfile = open(filename, 'rb')
        copy = pickle.load(dbfile)

        self.env = copy.env
        self.G = copy.G
        self.occupied_voxels = copy.occupied_voxels
        self.v_0 = copy.v_0
        self.R = copy.R
        self.current_q = copy.current_q
        self.vision_q = copy.vision_q
        self.final_executed = copy.final_executed
        self.object_poses = copy.object_poses
        dbfile.close()


def distance(vex1, vex2):
    """
    Helper function that returns the Euclidean distance between two configurations.
    It uses a "fudge" factor for the relationship between angles and distances.

    Args:
        vex1 (tuple): The first tuple
        vex2 (tuple): The second tuple
    Returns:
        float: The Euclidean distance between both tuples.
    """
    r = 0.01
    dist = 0
    for i in range(len(vex1)-1):
        dist += (vex1[i] - vex2[i])**2
    dist += (r*find_min_angle(vex1[2], vex2[2]))**2
    return dist**0.5

