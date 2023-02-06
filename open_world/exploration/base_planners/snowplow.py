import pickle
import datetime

from open_world.exploration.base_planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (joint_from_name, Pose,
                                                    Point, Euler, get_aabb_center, multiply, invert,
                                                    set_joint_positions, set_pose, get_pose)
import numpy as np
import time
from open_world.exploration.utils_graph import Graph
from open_world.exploration.utils import GRID_RESOLUTION, find_min_angle
from itertools import groupby


class Snowplow(Planner):

    def __init__(self, env, client=None):

        self.env = env
        self.client = client
        
        # Initializes a graph that contains the available movements
        self.G = Graph()
        self.G.initialize_full_graph(self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi / 8])

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
        self.current_q = None
        self.final_executed = []
        self.object_poses = None
        self.vision_q = dict()



    def get_plan(self, debug=False, loadfile=None, **kwargs):
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
        while not complete:
            # Save the movable boxes since they get changed during planning.
            mov_boxes_cache = list(self.env.movable_boxes)
            path = self.snowplow(self.current_q, q_goal, self.v_0)
            if path is None:
                return [key for key, _group in groupby(self.final_executed)]

            # Restore the current state of the movable boxes
            self.env.movable_boxes = list(mov_boxes_cache)
            self.current_q, complete, gained_vision, executed_path = self.execute_path(path)
            self.final_executed += executed_path
            self.v_0.update(gained_vision)

            # Ask for whether the user wants to save the current state to load it in the future.
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




    def snowplow(self, q_start, q_goal, v_0):
        path = self.a_star(q_start, q_goal, v_0)
        if path is None:
            print("Can't find a direct path to goal. Looking through movable.")
            path_relax = self.a_star(q_start, q_goal, v_0, ignore_movable=True)
            if path_relax is None:
                print("Can't find a path to goal")
                return None

            path = self.plan_clear_path(q_start, q_goal, path_relax, v_0)
            if path is None:
                print("Can't clear the obstacle")
                return None

        # Rearrange the formatting for the case of getting it from A*
        if len(path[0]) == 3:
            path = [(x, None) for x in path]
        return path

    def plan_clear_path(self, q_start, q_goal, p_through, v_0):
        obj_obstruction = self.env.find_path_movable_obstruction(p_through)
        attachment_poses = self.env.sample_attachment_poses(obj_obstruction, self.G)

        p_through_voxels = self.env.visibility_voxels_from_path(p_through)

        for attach_pose in attachment_poses:
            v = set(v_0)
            p_attach = self.a_star(q_start, attach_pose, v_0)
            if p_attach is None:
                continue

            v.update(self.env.get_optimistic_path_vision(p_attach, self.G))
            print("Ready to plan placement")
            self.env.remove_movable_object(obj_obstruction)
            max_samplings = 5
            i = 0
            while True:
                v_p = set(v)
                i += 1
                if i > max_samplings:
                    print("Max number of placement samples reached")
                    self.env.movable_boxes.append(obj_obstruction)
                    break
                q_place, grasp, obj = self.env.sample_placement(p_attach[-1], obj_obstruction, self.G,
                                                                p_through_voxels)
                if q_place is None:
                    print("Can't find placement. Retrying attachment")
                    self.env.movable_boxes.append(obj_obstruction)
                    break
                p_place = self.a_star(attach_pose, q_place, v_p, attachment=[obj_obstruction, grasp, obj])
                if p_place is None:
                    print("Can't find path to placement. Finding different placement")
                    continue
                v_p.update(self.env.get_optimistic_path_vision(p_place, self.G, attachment=[obj_obstruction, grasp, obj]))

                # After the clearing is done. Then find a path to the goal
                obj_oobb_placed = self.env.movable_object_oobb_from_q(obj_obstruction, p_place[-1], grasp)
                self.env.movable_boxes.append(obj_oobb_placed)
                p_goal = self.a_star(p_place[-1], q_goal, v_p)
                self.env.remove_movable_object(obj_oobb_placed)
                if p_goal is None:
                    print("Can't find path to goal after placement. Finding different placement")
                    continue

                self.env.movable_boxes.append(obj_obstruction)
                return [(x, None) for x in p_attach] + [(y, [obj_obstruction, grasp, obj]) for y in p_place] +\
                       [(x, None) for x in p_goal]
        self.env.movable_boxes.append(obj_obstruction)
        return None


    def action_fn(self, path, v_0, extended=None, ignore_movable=False, attachment=None):
        """
        Helper function to the search, that given a node, it gives all the possible actions to take with
        the inquired cost of each. Uses the vision constraint on each node based
        on the vision gained from the first path found to the node.

        Args:
            q (tuple): The node to expand.
            extended (set): Set of nodes that were already extended by the search.
        Returns:
            list: A list of available actions with the respective costs.
        """
        if extended is None:
            extended = set()
        actions = []
        q = path[-1]
        # Retrieve all the neighbors of the current node based on the graph of the space.
        for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
            q_prime = self.G.vertices[q_prime_i]

            # If the node has already been extended do not consider it.
            if q_prime in extended:
                continue

            # Check if there is an attached object that can only be pushed and prune actions
            # to depict this.
            if attachment is not None:
                if attachment[2] in self.env.push_only:
                    angle = round(np.arctan2(q_prime[1] - q[1], q_prime[0] - q[0]), 3)
                    if angle != q[2] or q_prime[2] != q[2]:
                        continue


            # Check for whether the new node is in obstruction with any obstacle.
            collisions, coll_objects = self.env.obstruction_from_path([q, q_prime], set(),
                                                                      ignore_movable=ignore_movable,
                                                                      attachment=attachment)
            if not collisions.shape[0] > 0 and (ignore_movable or coll_objects is None):
                if len(path) == 1:
                    v_q = v_0.union(self.env.get_optimistic_vision(q, self.G, attachment=attachment))
                else:
                    v_q = self.vision_q[path[-2]].union(self.env.get_optimistic_vision(q, self.G,
                                                                                       attachment=attachment))
                self.vision_q[q] = v_q
                s_q = self.env.visibility_voxels_from_path([q, q_prime], attachment=attachment)
                if s_q.issubset(v_q):
                    actions.append((q_prime, distance(q, q_prime)))
        return actions


    def a_star(self, q_start, q_goal, v_0, ignore_movable=False, attachment=None):
        """
        A* search algorithm.

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
            if np.linalg.norm(np.array(list(best_path[-1]))-np.array(list(q_goal)))<0.01:
                done = time.time() - current_t
                print("Extended nodes: {}".format(len(extended)))
                print("Search Time: {}".format(done))
                if self.debug:
                    self.G.plot_search(self.env, extended, path=best_path, goal=q_goal)
                return best_path

            extended.add(best_path[-1])
            actions = self.action_fn(best_path, v_0, extended=extended, ignore_movable=ignore_movable,
                                     attachment=attachment)
            for action in actions:
                paths.append((best_path + [action[0]], best_path_cost + action[1], distance(action[0], q_goal)))

            # TODO: Only sorting from heuristic. Faster but change if needed
            paths = sorted(paths, key=lambda x: x[-1] + x[-2], reverse=True)

        done = time.time() - current_t
        print("Extended nodes: {}".format(len(extended)))
        print("Search Time: {}".format(done))
        if self.debug:
            self.G.plot_search(self.env, extended, goal=q_goal)
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
        attachment = None
        for qi, node in enumerate(path):
            q, att = node
            if att is not None:
                obj_aabb = att[0].aabb
                obj = att[2]
            else:
                obj_aabb = None
                obj = None

            # Check if we are grasping a new object
            if attachment is None and obj is not None:
                coll_obj = self.env.get_movable_box_from_aabb(obj_aabb)
                # Compute the grasp transform of the attachment.
                base_pose = Pose(point=Point(x=q[0], y=q[1]), euler=Euler(yaw=q[2]), )
                # obj_pose = Pose(point=get_aabb_center(coll_obj.aabb))
                obj_pose = get_pose(obj)
                current_grasp = multiply(invert(base_pose), obj_pose)
                attachment = [coll_obj, current_grasp, obj]
                self.env.remove_movable_object(coll_obj)
            elif attachment is not None and obj is None:
                oobb = self.env.movable_object_oobb_from_q(attachment[0], path[qi - 1][0], attachment[1])
                self.env.movable_boxes.append(oobb)
                attachment = None

            # Check whether the next step goes into area that is unseen.
            next_occupied = self.env.visibility_voxels_from_path([q], attachment=attachment)
            for voxel in next_occupied:
                if self.env.visibility_grid.contains(voxel):
                    qi = qi - 1 if qi - 1 >= 0 else 0
                    print("Stepping into unseen area. Aborting")
                    # TODO: This holds only because we have two steps when attaching. Reaching the attachment
                    # then attaching
                    if attachment is not None:
                        oobb = self.env.movable_object_oobb_from_q(attachment[0], path[qi][0], attachment[1])
                        self.env.movable_boxes.append(oobb)
                    return path[qi][0], False, gained_vision, executed


            self.env.move_robot(q, self.joints, attachment=attachment)
            # Executed paths saved as a list of q and attachment.
            executed.append([q, attachment])

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(q, image_data)
            gained_vision.update(self.env.update_movable_boxes(image_data))
            gained_vision.update(self.env.update_visibility(camera_pose, image_data, q))

            # If an object is attached, do not detect it as an obstacle or a new movable object
            # TODO: Find a better method to clear the noise than the current one
            if attachment is not None:
                self.env.clear_noise_from_attached(q, attachment)

            # Check if remaining path is collision free under the new occupancy grid
            obstructions, collided_obj = self.find_obstruction_ahead(path[qi:], attachment)
            if obstructions.shape[0] > 0 or collided_obj is not None:
                print("Found a collision on this path. Aborting")
                self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                if attachment is not None:
                    oobb = self.env.movable_object_oobb_from_q(attachment[0], q, attachment[1])
                    self.env.movable_boxes.append(oobb)
                return q, False, gained_vision, executed
            self.env.plot_grids(visibility=True, occupancy=True, movable=True)

        return q, True, gained_vision, executed

    def find_obstruction_ahead(self, path, att):
        cached_movables = list(self.env.movable_boxes)
        if att is not None:
            oobb = self.env.movable_object_oobb_from_q(att[0], path[0][0], att[1])
            self.env.movable_boxes.append(oobb)

        attachment = None
        for qi, node in enumerate(path):
            q, att = node
            if att is not None:
                obj_aabb = self.env.movable_object_oobb_from_q(att[0], q, att[1]).aabb
                obj = att[2]
            else:
                obj_aabb = None
                obj = None
            # Check if we are grasping a new object
            if attachment is None and obj is not None:
                coll_obj = self.env.get_movable_box_from_aabb(obj_aabb)
                # Compute the grasp transform of the attachment.
                base_pose = Pose(point=Point(x=q[0], y=q[1]), euler=Euler(yaw=q[2]), )
                obj_pose = Pose(point=get_aabb_center(coll_obj.aabb))
                current_grasp = multiply(invert(base_pose), obj_pose)
                attachment = [coll_obj, current_grasp, obj]
                self.env.remove_movable_object(coll_obj)

            elif attachment is not None and obj is None:
                oobb = self.env.movable_object_oobb_from_q(attachment[0], path[qi - 1][0], attachment[1])
                self.env.movable_boxes.append(oobb)
                attachment = None

            obstructions, collided_obj = self.env.obstruction_from_path([q], set(),
                                                                        attachment=attachment)
            if obstructions.shape[0] > 0 or collided_obj is not None:
                self.env.movable_boxes = list(cached_movables)
                return obstructions, collided_obj

        self.env.movable_boxes = list(cached_movables)
        return obstructions, collided_obj




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
