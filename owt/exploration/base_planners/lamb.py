import datetime
import pickle
import time
from itertools import groupby

import numpy as np
import scipy.spatial

import owt.pb_utils as pbu
from owt.exploration.base_planners.planner import Planner
from owt.exploration.utils import GRID_RESOLUTION, find_min_angle
from owt.exploration.utils_graph import Graph

USE_COST = True


class Lamb(Planner):
    def __init__(self, env, client=None):
        # Sets up the environment and necessary data structures for the planner
        super(Lamb, self).__init__()

        self.env = env
        self.client = client

        # Initializes a graph that contains the available movements
        self.G = Graph()
        self.G.initialize_full_graph(
            self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi / 8]
        )

        # Creates a voxel structure that contains the vision space
        self.env.setup_default_vision(self.G)

        # Specific joints to move the robot in simulation
        self.joints = [
            pbu.joint_from_name(self.env.robot, "x", client=client),
            pbu.joint_from_name(self.env.robot, "y", client=client),
            pbu.joint_from_name(self.env.robot, "theta", client=client),
        ]

        # Structure used to save voxels that cannot be accessed by the robot, hence occupied
        self.occupied_voxels = dict()
        self.v_0 = None
        self.R = None
        self.complete = None
        self.current_q = None
        self.collided_object = None
        self.debug = False
        self.vision_q = dict()
        self.final_executed = []
        self.object_poses = None
        self.max_movables = 0

    def get_plan(self, loadfile=None, debug=False, **kwargs):
        """Creates a plan and executes it based on the given planner and
        environment.

        Args:
            loadfile (str): Location of the save file to load containing a previous state.
            debug (bool): Whether to enter debug mode
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

            pbu.set_joint_positions(
                self.env.robot, self.joints, self.current_q, client=self.client
            )
            for i, obj in enumerate(self.env.room.movable_obstacles):
                pbu.set_pose(obj, self.object_poses[i])
            self.env.plot_grids(True, True, True)

        complete = False
        # Continue looking for a plan until the robot has reached the goal.
        while not complete:
            # Save the movable boxes since they get changed during planning.
            self.max_movables = len(self.env.movable_boxes)
            mov_boxes_cache = list(self.env.movable_boxes)
            path = self.lamb(self.current_q, q_goal, self.v_0)
            # If at any point there is no possible path, then the search is ended.
            if path is None:
                return [key for key, _group in groupby(self.final_executed)]

            # Restore the current state of the movable boxes
            self.env.movable_boxes = list(mov_boxes_cache)
            # Execute path until it fails due to obstruction, or it reaches the goal. Update the
            # visibility based on what was observed while traversing the path.
            self.current_q, complete, gained_vision, executed_path = self.execute_path(
                path
            )
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

    def lamb(
        self,
        q_start,
        q_goal,
        v_0,
        attachment=None,
        obstructions=set(),
        enforced_obstacles=[],
        wrong_placements=set(),
        from_required=False,
        H=None,
    ):
        """Main function for path planning using VAMP.

        Args:
            q_start (tuple): Initial position from where to start to plan.
            q_goal (tuple): Goal position where the planning is ended.
            v_0 (set): Set of voxels which indicate what areas of the world have been seen before
                    initializing the planning
        Returns:
            list: A suggested plan for the robot to follow.
        """
        p = []
        v = set(v_0)
        q = q_start

        enf_obs = set()
        for oobb in enforced_obstacles:
            enf_obs = enf_obs.union(self.env.occupancy_grid.voxels_from_aabb(oobb.aabb))

        while True:
            # Find a path to goal, keeping the visualization constraint and return it if found
            p_final = self.a_star(
                q,
                q_goal,
                v,
                H=H,
                attachment=attachment,
                obstructions=obstructions,
                enforced_obstacles=enforced_obstacles,
                from_required=from_required,
            )
            if p_final is not None:
                p_final = [(x, attachment) for x in p_final]
                return p + p_final
            print("Couldn't find a direct path. Looking for a relaxed one")

            # If we are looking for visibility subgoals, just find directly through movable
            if from_required:
                p_relaxed = None
            else:
                # If a path to goal can't be found, find a relaxed path and use it as a subgoal
                p_relaxed = self.a_star(
                    q,
                    q_goal,
                    v,
                    H=H,
                    attachment=attachment,
                    relaxed=True,
                    obstructions=obstructions,
                    from_required=from_required,
                    enforced_obstacles=enforced_obstacles,
                )
            if p_relaxed is None:
                new_enforced = list(enforced_obstacles)
                p_move = None

                while True:
                    if attachment is not None:
                        if len(new_enforced) + 1 >= self.max_movables:
                            break
                    else:
                        if len(new_enforced) >= self.max_movables:
                            break

                    print("Can't find any path. Looking through movable obstacles")
                    p_through = self.a_star(
                        q,
                        q_goal,
                        v,
                        H=H,
                        attachment=attachment,
                        relaxed=True,
                        ignore_movable=True,
                        obstructions=obstructions,
                        enforced_obstacles=new_enforced,
                        from_required=from_required,
                    )
                    if p_through is None:
                        break

                    backchain_enforced = list(enforced_obstacles)
                    if attachment is not None:
                        backchain_enforced += [attachment[0]]

                    p_move = self.move_out_backchain(
                        q,
                        q_goal,
                        p_through,
                        v,
                        H=H,
                        attachment=attachment,
                        obstructions=obstructions,
                        enforced_obstacles=backchain_enforced,
                        wrong_placements=wrong_placements,
                        from_required=from_required,
                    )

                    if p_move is None:
                        print("Restricting which objects we can pass through")
                        obj_obstruction = self.env.find_path_movable_obstruction(
                            p_through, attachment=attachment
                        )
                        new_enforced.append(obj_obstruction)
                        continue
                    print("Found a path to clear the object. Continue the planning")
                    break

                if p_move is None:
                    print("Can't find any path. Aborting")
                    return None
                p += p_move
                v = v.union(
                    self.env.get_optimistic_path_vision(
                        p_move, self.G, obstructions=enf_obs
                    )
                )
                q = p_move[-1][0]
                continue

            new_enf = list(enforced_obstacles)
            if attachment is not None:
                new_enf += [attachment[0]]
            new_obs = set()
            for oobb in new_enf:
                new_obs = enf_obs.union(
                    self.env.occupancy_grid.voxels_from_aabb(oobb.aabb)
                )
            p_relaxed_voxels = self.env.visibility_voxels_from_path(
                p_relaxed, attachment=attachment
            )
            required_viewing = p_relaxed_voxels.difference(v)
            print("Planning to view a subgoal")

            while len(required_viewing) > 0:
                p_vis = self.vavp(
                    q,
                    required_viewing,
                    v,
                    obstructions=obstructions,
                    enforced_obstacles=new_enf,
                    wrong_placements=wrong_placements.union(p_relaxed_voxels),
                )
                # If the relaxed version fails, explore some environment. And restart the search
                if p_vis is None:
                    print("P_VIS failed. Observing some of the environment")
                    W = set(self.env.static_vis_grid.value_from_voxel.keys())
                    p_vis = self.tourist(
                        q,
                        W.difference(v),
                        v,
                        obstructions=obstructions,
                        enforced_obstacles=new_enf,
                        wrong_placements=wrong_placements.union(p_relaxed_voxels),
                    )
                if p_vis is None:
                    print("P_VIS failed again. Aborting")
                    return None
                p += p_vis
                v = v.union(
                    self.env.get_optimistic_path_vision(
                        p_vis, self.G, obstructions=new_obs
                    )
                )
                q = p_vis[-1][0]

                # Check if we made progress seen the area, if we did, then don't change the p_relaxed
                # Otherwise, find a new p_relaxed
                required_viewing_check = p_relaxed_voxels.difference(v)
                if len(required_viewing) != len(required_viewing_check):
                    required_viewing = required_viewing_check
                else:
                    p_relaxed = self.a_star(
                        q,
                        q_goal,
                        v,
                        H=H,
                        attachment=attachment,
                        relaxed=True,
                        obstructions=obstructions,
                        from_required=from_required,
                        enforced_obstacles=enforced_obstacles,
                    )
                    if p_relaxed is None:
                        break
                    p_relaxed_voxels = self.env.visibility_voxels_from_path(
                        p_relaxed, attachment=attachment
                    )
                    required_viewing = p_relaxed_voxels.difference(v)

            print("Planning to subgoal succeeded. Replanning final path")
            # Return to the starting configuration if we were moving an object
            if attachment is not None:
                p_return = self.a_star(
                    p[-1][0],
                    q_start,
                    v.union(
                        self.env.get_optimistic_path_vision(
                            p_vis, self.G, obstructions=new_obs
                        )
                    ),
                    obstructions=obstructions,
                    enforced_obstacles=enforced_obstacles + [attachment[0]],
                    from_required=from_required,
                )
                p_return = [(x, None) for x in p_return]
                p += p_return
                v = v.union(
                    self.env.get_optimistic_path_vision(
                        p_return, self.G, obstructions=new_obs
                    )
                )
                q = p_return[-1][0]

    def move_out_backchain(
        self,
        q_start,
        q_goal,
        p_through,
        v_0,
        H=None,
        attachment=None,
        obstructions=set(),
        enforced_obstacles=[],
        wrong_placements=set(),
        from_required=False,
    ):
        """Path planning using vamp for clearing an object out of the way.

        Args:
            q_start (tuple): Initial position from where to start to plan.
            p_through (list): The found path that goes through obstacles
            v_0 (set): Set of voxels which indicate what areas of the world have been seen before
                    initializing the planning
            obstructions (set): An extra set of obstructions to take into consideration.
        Returns:
            Whether we were able to attach successfully and the path through the object.
        """
        self.current_q = q_start

        new_obs = set()
        for oobb in enforced_obstacles:
            new_obs = new_obs.union(self.env.occupancy_grid.voxels_from_aabb(oobb.aabb))

        # Sample attachment poses
        obj_obstruction = self.env.find_path_movable_obstruction(
            p_through, attachment=attachment
        )
        voxels_from_obstruction = self.env.occupancy_grid.voxels_from_aabb(
            obj_obstruction.aabb
        )
        attachment_poses = self.env.sample_attachment_poses(
            obj_obstruction,
            self.G,
            obstructions=obstructions,
            enforced_obstacles=enforced_obstacles,
        )

        p_through_voxels = self.env.visibility_voxels_from_path(
            p_through, attachment=attachment
        )

        # Sort poses for a bias towards closer poses
        attachment_poses = sorted(attachment_poses, key=lambda x: distance(x, q_start))

        for attach_pose in attachment_poses:
            cached_movables = list(self.env.movable_boxes)
            v = set(v_0)
            p_attach = self.lamb(
                q_start,
                attach_pose,
                v,
                obstructions=obstructions,
                enforced_obstacles=enforced_obstacles + [obj_obstruction],
                wrong_placements=wrong_placements.union(p_through_voxels),
            )
            if p_attach is None:
                self.env.movable_boxes = cached_movables
                continue

            v.update(
                self.env.get_optimistic_path_vision(
                    p_attach, self.G, obstructions=new_obs
                )
            )
            print("Ready to plan placement")
            self.env.remove_movable_object(obj_obstruction)
            max_samplings = 5
            i = 0
            while True:
                v_p = set(v)
                cached_movables_p = list(self.env.movable_boxes)
                i += 1
                if i > max_samplings:
                    print("Max number of placement samples reached")
                    self.env.movable_boxes = cached_movables_p
                    self.env.movable_boxes.append(obj_obstruction)
                    break
                q_place, grasp, obj = self.env.sample_placement(
                    p_attach[-1][0],
                    obj_obstruction,
                    self.G,
                    p_through_voxels.union(wrong_placements),
                    enforced_obstacles=enforced_obstacles,
                )
                if q_place is None:
                    print("Can't find placement. Retrying attachment")
                    self.env.movable_boxes = cached_movables_p
                    self.env.movable_boxes.append(obj_obstruction)
                    break

                # Sub-procedure for placing
                p_place = self.lamb(
                    attach_pose,
                    q_place,
                    v_p,
                    attachment=[obj_obstruction, grasp, obj],
                    obstructions=obstructions,
                    wrong_placements=wrong_placements.union(p_through_voxels),
                    enforced_obstacles=enforced_obstacles,
                )
                if p_place is None:
                    self.env.movable_boxes = cached_movables_p
                    print("Can't find path to placement. Finding different placement")
                    continue
                v_p.update(
                    self.env.get_optimistic_path_vision(
                        p_place, self.G, obstructions=new_obs
                    )
                )

                # After the clearing is done then find a path to the goal, unless there is an object
                # attached
                obj_oobb_placed = self.env.movable_object_oobb_from_q(
                    obj_obstruction, p_place[-1][0], grasp
                )
                self.env.movable_boxes.append(obj_oobb_placed)
                if attachment is not None:
                    p_goal = self.lamb(
                        p_place[-1][0],
                        q_start,
                        v_p,
                        obstructions=obstructions,
                        enforced_obstacles=enforced_obstacles,
                        wrong_placements=wrong_placements,
                    )
                else:
                    p_goal = self.lamb(
                        p_place[-1][0],
                        q_goal,
                        v_p,
                        H=H,
                        obstructions=obstructions,
                        enforced_obstacles=enforced_obstacles,
                        wrong_placements=wrong_placements,
                        from_required=from_required,
                    )
                if p_goal is None:
                    self.env.movable_boxes = cached_movables_p
                    self.env.remove_movable_object(obj_oobb_placed)
                    print(
                        "Can't find path to goal after placement. Finding different placement"
                    )
                    continue

                return p_attach + p_place + p_goal
            self.env.movable_boxes = cached_movables
        self.env.movable_boxes = cached_movables
        return None

    def vavp(
        self, q, R, v, obstructions=set(), enforced_obstacles=[], wrong_placements=set()
    ):
        """Subprocedure to aid the planning on dividing the objective into
        subgoals and planning paths accordingly.

        Args:
            q (tuple): Current position of the robot.
            R (set) : Set of voxels that define the area we are interested in gaining vision from.
            v (set) : Set of tuples that define the already seen space.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: The suggested path that views some of the area of interest.
        """
        # Try visualizing the area of interest keeping the vision constraint.
        obstructions_new = obstructions.union(R)
        p_vis = self.tourist(
            q,
            R,
            v,
            obstructions=obstructions_new,
            enforced_obstacles=enforced_obstacles,
            wrong_placements=wrong_placements,
        )
        if p_vis is not None:
            return p_vis
        # If it can't view the area, find a relaxed path that does the same and make this new path
        # the new subgoal. Call the function recursively.
        # obstructions_new = obstructions.union(R)
        # p_relaxed = self.tourist(q, R, v, relaxed=True, obstructions=obstructions_new,
        #                          enforced_obstacles=enforced_obstacles, wrong_placements=wrong_placements)
        # if p_relaxed is not None:
        #     p_vis = self.vavp(q, self.env.visibility_voxels_from_path(p_relaxed).difference(v), v,
        #                       obstructions=obstructions_new, enforced_obstacles=enforced_obstacles,
        #                       wrong_placements=wrong_placements)
        #     if p_vis is not None:
        #         return p_vis
        return None

    def tourist(
        self,
        q_start,
        R,
        v_0,
        q_goal=None,
        relaxed=False,
        obstructions=set(),
        ignore_movable=False,
        enforced_obstacles=[],
        wrong_placements=set(),
    ):
        """Procedure used to find a path that partially or completely views
        some area of interest.

        Args:
            q_start (tuple): Starting position of the robot.
            R (set) : Set of voxels that define the area we are interested in gaining vision from.
            v_0 (set) : Set of tuples that define the already seen space.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
            ignore_movable (bool): Whether to ignore collisions with movable objects or not.
        Returns:
            list: The suggested path that views some area of interest.
        """
        new_obs = set()
        for oobb in enforced_obstacles:
            new_obs = new_obs.union(self.env.occupancy_grid.voxels_from_aabb(oobb.aabb))

        def heuristic_fn(q):
            # Previously used code that defines the heuristic as the smallest distance from the vision
            # gained in the configuration to the area of interest.
            vision_q = self.env.get_optimistic_vision(q, self.G, obstructions=new_obs)
            if len(R.intersection(vision_q)) != 0:
                return 0
            if len(vision_q) == 0:
                return (self.env.room.aabb.upper[0] - self.env.room.aabb.lower[0]) / 2
            s1 = np.array(list(vision_q))
            s2 = np.array(list(R))
            return scipy.spatial.distance.cdist(s1, s2).min() * GRID_RESOLUTION

        return self.lamb(
            q_start,
            q_goal,
            v_0,
            H=heuristic_fn,
            obstructions=obstructions,
            enforced_obstacles=enforced_obstacles,
            wrong_placements=wrong_placements,
            from_required=True,
        )

    def action_fn(
        self,
        path,
        v_0,
        relaxed=False,
        extended=None,
        obstructions=None,
        ignore_movable=False,
        attachment=None,
        enforced_obstacles=None,
    ):
        """Helper function to the search, that given a node, it gives all the
        possible actions to take with the inquired cost of each. Uses the
        vision constraint on each node based on the vision gained from the
        first path found to the node.

        Args:
            path (list): The path obtained to reach the current node on the search.
            v_0 (set): Set of tuples that define the already seen space.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            extended (set): Set of nodes that were already extended by the search.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
            ignore_movable (bool): Whether to ignore collisions with movable objects or not.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            list: A list of available actions with the respective costs.
        """
        if extended is None:
            extended = set()
        if obstructions is None:
            obstructions = set()
        if enforced_obstacles is None:
            enforced_obstacles = []
        actions = []
        q = path[-1]
        new_obs = set()
        for oobb in enforced_obstacles:
            new_obs = new_obs.union(self.env.occupancy_grid.voxels_from_aabb(oobb.aabb))

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
                    angle = np.arctan2(q_prime[1] - q[1], q_prime[0] - q[0])
                    angle = (
                        round(angle + 2 * np.pi, 3) if angle < 0 else round(angle, 3)
                    )
                    if angle != q[2] or q_prime[2] != q[2]:
                        continue

            if relaxed:
                # Check for whether the new node is in obstruction with any obstacle.
                collisions, coll_objects = self.env.obstruction_from_path(
                    [q, q_prime],
                    obstructions,
                    ignore_movable=ignore_movable,
                    attachment=attachment,
                    enforced_obstacles=enforced_obstacles,
                )
                if not collisions.shape[0] > 0 and (
                    ignore_movable or coll_objects is None
                ):
                    if len(path) == 1:
                        v_q = v_0.union(
                            self.env.get_optimistic_vision(
                                q, self.G, attachment=attachment, obstructions=new_obs
                            )
                        )
                    else:
                        v_q = self.vision_q[path[-2]].union(
                            self.env.get_optimistic_vision(
                                q, self.G, attachment=attachment, obstructions=new_obs
                            )
                        )
                    self.vision_q[q] = v_q
                    s_q = self.env.visibility_voxels_from_path(
                        [q, q_prime], attachment=attachment
                    )
                    # If the node follows the visibility constraint, add it normally.
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
                    # If it does not follow the visibility constraint, add it with a special cost.
                    else:
                        # cost = distance(q, q_prime) *\
                        #         abs(self.volume_from_voxels(self.env.static_vis_grid, s_q.difference(v_q)))
                        cost = distance(q, q_prime) * (len(s_q.difference(v_q)) + 1)
                        actions.append((q_prime, cost))
            else:
                # In the not relaxed case only add nodes when the visibility constraint holds.
                collisions, coll_objects = self.env.obstruction_from_path(
                    [q, q_prime],
                    obstructions,
                    ignore_movable=ignore_movable,
                    attachment=attachment,
                    enforced_obstacles=enforced_obstacles,
                )
                if not collisions.shape[0] > 0 and (
                    ignore_movable or coll_objects is None
                ):
                    # s_q = self.env.visibility_points_from_path([q, q_prime], attachment=attachment)
                    # if self.env.in_view_cone(s_q, path):
                    #    actions.append((q_prime, distance(q, q_prime)))
                    if len(path) == 1:
                        v_q = v_0.union(
                            self.env.get_optimistic_vision(
                                q, self.G, attachment=attachment, obstructions=new_obs
                            )
                        )
                    else:
                        v_q = self.vision_q[path[-2]].union(
                            self.env.get_optimistic_vision(
                                q, self.G, attachment=attachment, obstructions=new_obs
                            )
                        )
                    self.vision_q[q] = v_q
                    s_q = self.env.visibility_voxels_from_path(
                        [q, q_prime], attachment=attachment
                    )
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))

        return actions

    @staticmethod
    def volume_from_voxels(grid, voxels):
        """Calculates the volume of a given set of voxels.

        Args:
            grid (object): The grid to which the voxels belong to.
            voxels (set): The set of voxels from which to determine the volume
        Returns:
            float: The volume of the voxels.
        """
        if len(voxels) == 0:
            return 0
        voxel_vol = pbu.get_aabb_volume(grid.aabb_from_voxel(next(iter(voxels))))
        return voxel_vol * len(voxels)

    def a_star(
        self,
        q_start,
        q_goal,
        v_0,
        H=None,
        relaxed=False,
        obstructions=set(),
        ignore_movable=False,
        attachment=None,
        enforced_obstacles=[],
        from_required=False,
    ):
        """A* search algorithm.

        Args:
            q_start (tuple): Start node.
            q_goal (tuple): Goal node.
            v_0 (set): Set of tuples that define the already seen space.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
            ignore_movable (bool): Whether to ignore collisions with movable objects or not.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            list: The path from start to goal.
        """
        if H is None or not from_required:
            H = lambda x: distance(x, q_goal)

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
                    if (
                        np.linalg.norm(
                            np.array(list(best_path[-1])) - np.array(list(q_goal))
                        )
                        < 0.01
                    ):
                        done = time.time() - current_t
                        print("Extended nodes: {}".format(len(extended)))
                        print("Search Time: {}".format(done))
                        if self.debug:
                            self.G.plot_search(
                                self.env,
                                extended,
                                path=best_path,
                                goal=q_goal,
                                enforced_obstacles=enforced_obstacles,
                                R=obstructions,
                            )
                        return best_path

                else:
                    if H(best_path[-1]) == 0:
                        q_goal = best_path[-1]
                        done = time.time() - current_t
                        print("Extended nodes: {}".format(len(extended)))
                        print("Search Time: {}".format(done))
                        if self.debug:
                            self.G.plot_search(
                                self.env,
                                extended,
                                path=best_path,
                                goal=q_goal,
                                enforced_obstacles=enforced_obstacles,
                                R=obstructions,
                            )
                        return best_path

                extended.add(best_path[-1])
                actions = self.action_fn(
                    best_path,
                    v_0,
                    relaxed=relaxed,
                    extended=extended,
                    obstructions=obstructions,
                    ignore_movable=ignore_movable,
                    attachment=attachment,
                    enforced_obstacles=enforced_obstacles,
                )
                for action in actions:
                    paths.append(
                        (
                            best_path + [action[0]],
                            best_path_cost + action[1],
                            H(action[0]),
                        )
                    )

                # Only sorting from heuristic. Faster but change if needed
                if USE_COST:
                    paths = sorted(paths, key=lambda x: x[-1] + x[-2], reverse=True)
                else:
                    paths = sorted(paths, key=lambda x: x[-1], reverse=True)

            done = time.time() - current_t
            print("Extended nodes: {}".format(len(extended)))
            print("Search Time: {}".format(done))
            if self.debug:
                self.G.plot_search(
                    self.env,
                    extended,
                    goal=q_goal,
                    enforced_obstacles=enforced_obstacles,
                    R=obstructions,
                )
            return None
        except KeyboardInterrupt:
            return None

    def execute_path(self, path):
        """Executes a given path in simulation until it is complete or no
        longer feasible.

        Args:
            path (list): The path to execute.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            tuple: A tuple containing the state where execution stopped, whether it was able to reach the goal,
             the gained vision, and the collided movable object.
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
                base_pose = pbu.Pose(
                    point=pbu.Point(x=q[0], y=q[1]),
                    euler=pbu.Euler(yaw=q[2]),
                )
                obj_pose = pbu.get_pose(obj)
                current_grasp = pbu.multiply(pbu.invert(base_pose), obj_pose)
                attachment = [coll_obj, current_grasp, obj]
                self.env.remove_movable_object(coll_obj)
            elif attachment is not None and obj is None:
                oobb = self.env.movable_object_oobb_from_q(
                    attachment[0], path[qi - 1][0], attachment[1]
                )
                self.env.movable_boxes.append(oobb)
                attachment = None

            # Check whether the next step goes into area that is unseen.
            next_occupied = self.env.visibility_voxels_from_path(
                [q], attachment=attachment
            )
            for voxel in next_occupied:
                if self.env.visibility_grid.contains(voxel):
                    qi = qi - 1 if qi - 1 >= 0 else 0
                    print("Stepping into unseen area. Aborting")
                    # TODO: This holds only because we have two steps when attaching. Reaching the attachment
                    # then attaching
                    if attachment is not None:
                        oobb = self.env.movable_object_oobb_from_q(
                            attachment[0], path[qi][0], attachment[1]
                        )
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
            obstructions, collided_obj = self.find_obstruction_ahead(
                path[qi:], attachment
            )
            if obstructions.shape[0] > 0 or collided_obj is not None:
                print("Found a collision on this path. Aborting")
                self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                if attachment is not None:
                    oobb = self.env.movable_object_oobb_from_q(
                        attachment[0], q, attachment[1]
                    )
                    self.env.movable_boxes.append(oobb)
                # If the current q is in collision with movable due to expansion of aabb in
                # last step or an intermediate configuration. Then backtrack to the
                # last step on the already executed path where this does not happen.
                _, collided_obj = self.find_obstruction_ahead([(q, None)], None)
                while collided_obj is not None:
                    qi -= 1
                    q = path[qi][0]
                    if attachment is not None:
                        self.env.remove_movable_object(oobb)

                    self.env.move_robot(q, self.joints, attachment=attachment)
                    executed.append([q, attachment])

                    if attachment is not None:
                        oobb = self.env.movable_object_oobb_from_q(
                            attachment[0], q, attachment[1]
                        )
                        self.env.movable_boxes.append(oobb)
                    _, collided_obj = self.find_obstruction_ahead([(q, None)], None)

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
                base_pose = pbu.Pose(
                    point=pbu.Point(x=q[0], y=q[1]),
                    euler=pbu.Euler(yaw=q[2]),
                )
                obj_pose = pbu.Pose(point=pbu.get_aabb_center(coll_obj.aabb))
                current_grasp = pbu.multiply(pbu.invert(base_pose), obj_pose)
                attachment = [coll_obj, current_grasp, obj]
                self.env.remove_movable_object(coll_obj)

            elif attachment is not None and obj is None:
                oobb = self.env.movable_object_oobb_from_q(
                    attachment[0], path[qi - 1][0], attachment[1]
                )
                self.env.movable_boxes.append(oobb)
                attachment = None

            obstructions, collided_obj = self.env.obstruction_from_path(
                [q], set(), attachment=attachment
            )
            if obstructions.shape[0] > 0 or collided_obj is not None:
                self.env.movable_boxes = list(cached_movables)
                return obstructions, collided_obj

        self.env.movable_boxes = list(cached_movables)
        return obstructions, collided_obj

    def save_state(self):
        """Saves the current state of the Vamp planning algorithm."""
        current_time = datetime.datetime.now()
        dbfile = open(
            "saves/{}_state_{}_{}_{}_{}_{}_{}.dat".format(
                self.env.__class__.__name__,
                self.__class__.__name__,
                current_time.month,
                current_time.day,
                current_time.hour,
                current_time.minute,
                current_time.second,
            ),
            "wb",
        )
        pickle.dump(self, dbfile)
        dbfile.close()

    def load_state(self, filename):
        """Loads the specified file containing a state of the Vamp planner.

        Args:
            filename (str): The path to the file to load.
        """
        dbfile = open(filename, "rb")
        copy = pickle.load(dbfile)

        self.env = copy.env
        self.G = copy.G
        self.occupied_voxels = copy.occupied_voxels
        self.v_0 = copy.v_0
        self.R = copy.R
        self.current_q = copy.current_q
        self.collided_object = copy.collided_object
        self.vision_q = copy.vision_q
        self.final_executed = copy.final_executed
        self.object_poses = copy.object_poses
        dbfile.close()


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
    R = 0.01
    dist = 0
    for i in range(len(vex1) - 1):
        dist += (vex1[i] - vex2[i]) ** 2
    dist += (R * find_min_angle(vex1[2], vex2[2])) ** 2
    return dist**0.5
