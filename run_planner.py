#!/usr/bin/env python3

from __future__ import print_function
import sys
import argparse
import warnings

sys.path.extend(["tamp","pybullet_planning"])
warnings.filterwarnings("ignore")

from itertools import product

from pybullet_tools.utils import (load_pybullet, connect, wait_for_user, wait_if_gui)

from open_world.planning.streams import GEOMETRIC_MODES, LEARNED_MODES, MODE_ORDERS
from open_world.simulation.policy import Policy
from open_world.simulation.tasks import GOALS, task_from_goal

from open_world.exploration.base_planners.a_star_search import AStarSearch
from open_world.exploration.base_planners.snowplow import Snowplow
from open_world.exploration.base_planners.namo import Namo
from open_world.exploration.base_planners.rrt import RRT
from open_world.exploration.base_planners.lamb import Lamb

from open_world.nlp.speech_to_goal import get_goal_audio
from open_world.nlp.text_to_goal import text_to_goal

from robots.movo.movo_utils import MOVO_PATH, MovoRobot
from robots.movo.movo_worlds import movo_world_from_problem
from robots.panda.panda_utils import PANDA_PATH, PandaRobot
from robots.panda.panda_worlds import panda_world_from_problem
from robots.pr2.pr2_utils import PR2_PATH, PR2Robot
from robots.pr2.pr2_worlds import pr2_world_from_problem

ROBOTS = ["pr2", "panda", "movo"]
SEG_MODELS = ["maskrcnn", "uois", "ucn", "all"]
SHAPE_MODELS = ["msn", "atlas"]

robot_paths = {"pr2": PR2_PATH, "panda": PANDA_PATH, "movo": MOVO_PATH}
robot_entities = {"pr2": PR2Robot, "panda": PandaRobot, "movo": MovoRobot}

robot_simulated_worlds = {
    "pr2": pr2_world_from_problem,
    "panda": panda_world_from_problem,
    "movo": movo_world_from_problem,
}

base_planners = {"snowplow": Snowplow,
                 "astar": AStarSearch,
                 "namo": Namo,
                 "rrt": RRT,
                 "lamb": Lamb}

GRASP_MODES = GEOMETRIC_MODES + [
    mode + order for mode, order in product(LEARNED_MODES, MODE_ORDERS)
]


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--debug", action="store_true", help="")
    parser.add_argument(
        "-o",
        "--observable",
        action="store_true",
        help="Uses the groundtruth PyBullet objects as the estimated objects.",
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Saves the RGD, depth, and segmented images.",
    )

    parser.add_argument(
        "-v", "--viewer", action="store_true", help="Enables the PyBullet viewer."
    )

    parser.add_argument(
        "-c",
        "--client",
        type=int,
        default=0,
        help="Selects the client physics engine to view when in viewer mode",
    )

    parser.add_argument(
        "-rc",
        "--real_camera",
        action="store_true",
        help="Use a realsense camera for perception",
        default=False,
    )

    parser.add_argument(
        "-re",
        "--real_execute",
        action="store_true",
        help="Execute the positions commands on a real robot",
        default=False,
    )

    # shape completion
    parser.add_argument(
        "-sc",
        "--shape_completion",
        action="store_true",
        help="Uses a DNN for shape completion.",
    )

    parser.add_argument(
        "-scm",
        "--shape_completion_model",
        type=str,
        default="msn",
        choices=SHAPE_MODELS,
        help="Selects the DNN shape completion model.",
    )

    parser.add_argument(
        "-convex",
        action="store_false",
        help="Uses convex hulls instead of concave hulls to estimate objects.",
    )

    parser.add_argument("-disable_project", action="store_true")

    # segmentation
    parser.add_argument(
        "-seg",
        "--segmentation",
        action="store_true",
        help="Uses a DNN for segmentation.",
    )

    parser.add_argument(
        "-rgbd", "--maskrcnn_rgbd", action="store_true", help="Uses RGBD for maskrcnn."
    )
    parser.add_argument(
        "-segm",
        "--segmentation_model",
        type=str,
        default="ucn",
        choices=SEG_MODELS,
        help="Selects the DDN model for segmentation",
    )
    
    parser.add_argument(
        "-det",
        "--fasterrcnn_detection",
        action="store_true",
        help="Uses FasterRCNN to label any cup or bowl instances that were segmented by UOIS.",
    )

    # grasping
    parser.add_argument(
        "-g",
        "--grasp_mode",
        type=str,
        default="mesh",
        choices=GRASP_MODES,
        help="Selects the grasp generation strategy.",
    )

    # task
    parser.add_argument("-p", "--goal", default="all_green", help="Specifies the task.")
    parser.add_argument("-w", "--world", default="problem0", help="Specifies the task.")

    # exploration    
    parser.add_argument("-exp", "--exploration", action="store_true", help="Use exploration prior to running m0m")
    parser.add_argument("-bp", "--base_planner", default="lamb", help="Specifies the planner to use for base navigation")

    # robot
    parser.add_argument("-r", "--robot", default="pr2", help="Specifies the robot.")

    # interactive goals
    parser.add_argument("-ti", "--text_interactive",  action="store_true", help="Use text input to specify the goal")
    parser.add_argument("-vi", "--voice_interactive",  action="store_true", help="Use audio input to specify the goal")

    return parser


def setup_robot_pybullet(args):
    connect(use_gui=args.viewer)
    robot_body = load_pybullet(robot_paths[args.robot], fixed_base=True, client=None)
    return robot_body, None

def get_task(args):
    problem_from_name = {fn.__name__: fn for fn in GOALS}
    if(args.voice_interactive):
        return task_from_goal(args, get_goal_audio())
    elif(args.text_interactive):
        goal, _ = text_to_goal(wait_for_user("Enter a command: \n"))
        return task_from_goal(args, goal)
    else:    
        if args.goal not in problem_from_name:
            raise ValueError(args.goal)
        problem_fn = problem_from_name[args.goal]
        task = problem_fn(args)
        task.name = args.goal
        return task


def main():

    # Parse the args
    parser = create_parser()
    args = parser.parse_args()

    # Create the robot
    robot_body, client = setup_robot_pybullet(args)

    robot = robot_entities[args.robot](robot_body, 
                                       client=client, 
                                       real_execute = args.real_execute, 
                                       real_camera = args.real_camera)

    # Set up the world run the task
    real_world = robot_simulated_worlds[args.robot](
        args.world, robot, args, client=client
    )
    policy = Policy(args, robot, known=real_world.known, client=client)
    done = False
    while(not done): 
        wait_if_gui("Press enter")
        task = get_task(args)
        if(args.exploration):
            policy.run_exploration(task, real_world=real_world, client=client, \
                room = real_world.room, base_planner=base_planners[args.base_planner])
        else:
            policy.run(task, real_world=real_world, client=client)

        done = not args.voice_interactive and not args.text_interactive

if __name__ == "__main__":
    main()