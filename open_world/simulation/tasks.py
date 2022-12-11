# Currently there are only simulated tasks for pr2


from pddlstream.utils import str_from_object
from pybullet_tools.pr2_utils import ARM_NAMES
from pybullet_tools.utils import get_pairs
from open_world.planning.planner import (
    PARAM,
    And,
    Exists,
    ForAll,
    Imply,
    On
)


SKILLS = ["pick", "push"]

PARAM1 = PARAM
PARAM2 = "?o2"
PARAM3 = "?o3"

DEFAULT_SKILLS = ["pick"]
COLORS = ["red", "green", "blue", "yellow"]
CATEGORIES = ["mustard_bottle"]
BOWL = "bowl"
DEPTH_SCALE = 3.0
NUMS = list(range(2, 3 + 1))


class Task(object):
    def __init__(
        self,
        name=None,
        init=[],
        goal_parts=[],
        assume={},
        arms=ARM_NAMES,
        skills=SKILLS,
        return_init=True,
        empty_arms=False,
        goal_regions=[],
    ):
        # TODO: args?
        self.name = name
        self.init = tuple(init)
        self.goal_parts = tuple(goal_parts)
        self.assume = dict(assume)
        self.arms = tuple(arms)
        self.skills = tuple(skills)
        self.return_init = return_init
        self.empty_arms = empty_arms
        self.goal_regions = tuple(goal_regions)

    def __repr__(self):
        return "{}{}".format(self.__class__.__name__, str_from_object(self.__dict__))


def task_from_goal(args, goal):
    task = Task(
        goal_parts=[goal],
        arms=args.arms,
        skills=DEFAULT_SKILLS,
    )
    return task

def holding(args):  # For testing grasping
    task = Task(
        goal_parts=[
            Exists(
                [PARAM1],
                And(
                    ("Graspable", PARAM1),
                    ("Holding", PARAM1),
                ),
            ),
        ],
        arms=args.arms,
        skills=DEFAULT_SKILLS,
    )
    return task


def all_bowl(args):
    # All objects in the bowl that is closet to their color
    return Task(
        goal_parts=[
            ForAll(
                [PARAM1],
                Imply(
                    ("Graspable", PARAM1),  # Not(('Category', PARAM1, BOWL)),
                    Exists(
                        [PARAM2],
                        And(
                            ("Category", PARAM2, BOWL),
                            ("ClosestColor", PARAM2, PARAM1),
                            ("In", PARAM1, PARAM2),
                        ),
                    ),
                ),
            ),
        ],
        assume={
            "category": [BOWL],
        },
        arms=args.arms,
        skills=DEFAULT_SKILLS,
    )


GOALS = [
    holding,
    all_bowl,
]

##################################################

for color in COLORS:

    # Object on a region of a particular color
    GOALS.append(
        lambda args, color=color: Task(
            goal_parts=[
                Exists(
                    [PARAM1, PARAM2],
                    And(("Graspable", PARAM1), ("Color", PARAM2, color), On(PARAM2)),
                ),
            ],
            arms=args.arms,
            skills=DEFAULT_SKILLS,
        )
    )
    GOALS[-1].__name__ = f"exists_{color}"  # .format(color)

    # Category on a region of a particular color
    category = "potted_meat_can"  # mustard_bottle | power_drill
    GOALS.append(
        lambda args, color=color: Task(
            goal_parts=[
                Exists(
                    [PARAM1, PARAM2],
                    And(
                        ("Category", PARAM1, category),
                        ("Color", PARAM2, color),
                        On(PARAM2),
                    ),
                ),
            ],
            assume={"category": [category]},
            arms=args.arms,
            skills=DEFAULT_SKILLS,
        )
    )  #'color': ['red],
    GOALS[-1].__name__ = f"{category}_{color}"  # .format(color)

    # Object closet to color on region
    GOALS.append(
        lambda args, color=color: Task(
            goal_parts=[
                Exists(
                    [PARAM1, PARAM2],
                    And(
                        ("Graspable", PARAM1),
                        ("ClosestColor", PARAM1, color),
                        ("Region", PARAM2),
                        On(PARAM2),
                    ),
                ),
            ],
            assume={"category": [category]},
            arms=args.arms,
            skills=DEFAULT_SKILLS,
        )
    )
    GOALS[-1].__name__ = f"closest_{color}" 

    # Object of a particular color on a region
    GOALS.append(
        lambda args, color=color: Task(
            goal_parts=[
                Exists(
                    [PARAM1, PARAM2],
                    And(("Graspable", PARAM1), ("Color", PARAM2, color), On(PARAM2)),
                ),
            ],
            arms=args.arms,
            skills=DEFAULT_SKILLS,
        )
    )
    GOALS[-1].__name__ = f"exists_{color}"  # .format(color)

    # All on a region of a particular color
    GOALS.append(
        lambda args, color=color: Task(
            goal_parts=[
                ForAll(
                    [PARAM1],
                    Imply(
                        ("Graspable", PARAM1),
                        Exists(
                            [PARAM2],
                            And(
                                ("Region", PARAM2), ("Color", PARAM2, color), On(PARAM2)
                            ),
                        ),
                    ),
                ),
            ],
            arms=args.arms,
            skills=DEFAULT_SKILLS,
        )
    )
    GOALS[-1].__name__ = f"all_{color}"

#########################

for num in NUMS:
    # TODO: all stacked

    # Tower of num objects
    parameters = ["?o{}".format(i + 1) for i in range(num)]
    conditions = [("Graspable", param) for param in parameters] + [
        ("On", param1, param2)
        for param1, param2, in get_pairs(parameters)
    ]  # Graspable | Movable
    GOALS.append(
        lambda args, num=num: Task(
            goal_parts=[
                Exists(parameters, And(*conditions)),
            ],
            arms=args.arms,
            skills=DEFAULT_SKILLS,
        )
    )
    GOALS[-1].__name__ = f"stack_{num}"
