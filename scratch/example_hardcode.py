#!/usr/bin/env python3

from __future__ import print_function

import sys
import warnings

import numpy as np
import pybullet as p

warnings.filterwarnings("ignore")
sys.path.extend(
    [
        "pddlstream",
        "pybullet-planning",
    ]
)
import itertools
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import open3d
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
from movo.movo_utils import MOVO_PATH, MovoPolicy, MovoRobot
from movo.movo_worlds import movo_world_from_problem
from pddlstream.algorithms.algorithm import reset_globals
from pddlstream.algorithms.meta import analyze_goal
from pddlstream.algorithms.serialized import solve_all_goals, solve_next_goal
from pddlstream.language.constants import (NOT, And, Equal, Evaluation, Exists,
                                           Fact, ForAll, Head, Imply, Not,
                                           PDDLProblem, Solution, get_args,
                                           get_prefix, is_head, is_plan,
                                           print_solution)
from pddlstream.language.conversion import replace_expression
from pddlstream.language.external import never_defer
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.language.stream import StreamInfo
from pddlstream.utils import Profiler, get_file_path, lowercase, read
from pybullet_tools.pr2_utils import ARM_NAMES, LEFT_ARM, set_group_positions
from pybullet_tools.utils import (AABB, BLUE, PI, CameraImage, Point, Pose,
                                  connect, disconnect, get_pairs, invert,
                                  load_pybullet, pixel_from_point, read,
                                  tform_point, wait_if_gui)
from pybullet_tools.voxels import VoxelGrid
from sklearn.metrics import pairwise_distances_argmin_min

from owt.estimation.belief import (Belief, EstimatedObject, downsample_cluster,
                                   is_object_label, iterate_image)
from owt.estimation.dnn import str_from_int_seg_general
from owt.estimation.geometry import cloud_from_depth, estimate_surface_mesh
from owt.estimation.observation import (extract_point, image_from_labeled,
                                        save_camera_images)
from owt.planning.planner import iterate_sequence
from owt.planning.primitives import (GroupConf, RelativePose, Sequence,
                                     WorldState)
from owt.planning.streams import (get_grasp_gen_fn, get_plan_motion_fn,
                                  get_plan_pick_fn)
from owt.simulation.tasks import GOALS
from run_planner import (create_parser, robot_entities, robot_simulated_worlds,
                         setup_robot_pybullet)

SAVE_DIR = "temp_graphs/"


def reset_robot(robot):
    conf = robot.get_default_conf()
    for group, positions in conf.items():
        robot.set_group_positions(group, positions)


# Parse the args
problem_from_name = {fn.__name__: fn for fn in GOALS}
parser = create_parser()
args = parser.parse_args()

# Create the robot
robot_body = setup_robot_pybullet(args)
robot = robot_entities[args.robot](robot_body, args=args)
reset_robot(robot)


# Create the task
if args.goal not in problem_from_name:
    raise ValueError(args.goal)
problem_fn = problem_from_name[args.goal]
task = problem_fn(args)

# Create the simulated world
real_world = robot_simulated_worlds[args.robot](args.world, robot, args)
state = WorldState()

# Create the stream sampler functions
motion_planner = get_plan_motion_fn(robot, environment=[])
pick_planner = get_plan_pick_fn(robot)
grasp_finder = get_grasp_gen_fn(robot, [], grasp_mode="top")

# Set up sampler inputs
init_confs = {group: GroupConf(robot, group, important=True) for group in robot.groups}
arm = "main_arm"
obj = real_world.movable[0]
pose = RelativePose(obj)
base_conf = init_confs["base"]
group = "main_arm"
q1 = init_confs[group]


# Get a grasp
grasp = next(grasp_finder(arm, obj))

# Use the grasp to get a pick motion
pick = pick_planner(arm, obj, pose, grasp[0], base_conf)
q2, at = pick

# Use the resulting conf to motion plan
motion_plan = motion_planner(group, q1, q2)


# Reset state to original
robot.remove_components()
p.removeAllUserDebugItems()
state.assign()

# Execute the plan
iterate_sequence(state, Sequence([motion_plan[0], at]))
