##################################################

# TODO: specify the available target regions

PARAM1 = PARAM
PARAM2 = "?o2"
PARAM3 = "?o3"

DEFAULT_SKILLS = ["pick"]

COLORS = ["red", "green", "blue", "yellow"]
CATEGORIES = ["mustard_bottle"]
NUMS = list(range(2, 3 + 1))


def holding(args):  # For testing grasping
    # Holding object
    task = Task(
        goal_parts=[
            Exists(
                [PARAM1],
                And(
                    ("Graspable", PARAM1),
                    ("Holding", PARAM1),
                    # ('ArmHolding', arm_from_side(args.arms[0]), PARAM1),
                ),
            ),
            # Exists([PARAM2], And(('Graspable', PARAM2), ('ArmHolding', arm_from_side(args.arms[-1]), PARAM2))),
            # HoldingCategory(category=args.ycb), # Remove HandEmpty reset condition
            # CategoryOn(category=args.ycb, surface=region),
            # AllCategoryOn(category=args.ycb, surface=region),
        ],
        arms=args.arms,
        skills=DEFAULT_SKILLS,
        return_init=False,
        empty_arms=False,
    )
    # assume = {'category' : [args.ycb]}

    return task


def holding_category(category):
    # Holding object of a particular category
    value_from_arg = locals()
    # print(inspect.getfullargspec(all_color))
    # print(inspect.signature(all_color))
    def fn(args):
        return Task(
            goal_parts=[
                ExistsObject(
                    ("Category", PARAM1, category),
                    # ('ArmHolding', arm_from_side(args.arms[0]), PARAM1),
                    ("Holding", PARAM1),
                ),
            ],
            assume={
                "category": [category],
            },
            arms=args.arms,
            skills=DEFAULT_SKILLS,
        )

    for arg, value in value_from_arg.items():
        fn.__name__ = holding_category.__name__.replace(arg, value)
    return fn


def exists_bowl(args):
    # Object in a bowl
    return Task(
        goal_parts=[
            Exists(
                [PARAM1, PARAM2],
                And(
                    ("Graspable", PARAM1),
                    ("Category", PARAM2, BOWL),
                    Not(Equal(PARAM1, PARAM2)),  # TODO: not working
                    ("In", PARAM1, PARAM2),
                ),
            ),
        ],
        assume={
            "category": [BOWL],
        },
        arms=args.arms,
        skills=DEFAULT_SKILLS,
    )


def all_bowl(args):
    # All objects in the bowl that is closet to their color
    return Task(
        goal_parts=[
            # ForAll([PARAM1, PARAM2], Imply(
            #     And(('Graspable', PARAM1), ('Category', PARAM2, BOWL), Not(Equal(PARAM1, PARAM2))),
            #     ('In', PARAM1, PARAM2))),
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


PROBLEMS = [
    # TODO: integrate with the simulated tasks
    holding,
    exists_bowl,
    all_bowl,
] + [holding_category(category) for category in CATEGORIES]

##################################################

for color in COLORS:

    # Object on a region of a particular color
    PROBLEMS.append(
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
    PROBLEMS[-1].__name__ = f"exists_{color}"  # .format(color)

    # Category on a region of a particular color
    category = "power_drill"  # mustard_bottle | power_drill
    PROBLEMS.append(
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
    PROBLEMS[-1].__name__ = f"mustard_{color}"  # .format(color)

    # Object closet to color on region
    PROBLEMS.append(
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
    PROBLEMS[-1].__name__ = f"closest_{color}"  # .format(color)

    # Object of a particular color on a region
    PROBLEMS.append(
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
    PROBLEMS[-1].__name__ = f"exists_{color}"  # .format(color)

    # All on a region of a particular color
    PROBLEMS.append(
        lambda args, color=color: Task(
            goal_parts=[
                ForAll(
                    [PARAM1],
                    Imply(
                        ("Graspable", PARAM1),  # Not(('Category', PARAM1, BOWL)),
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
    PROBLEMS[-1].__name__ = f"all_{color}"

#########################

for num in NUMS:
    # TODO: all stacked

    # Tower of num objects
    parameters = ["?o{}".format(i + 1) for i in range(num)]
    conditions = [("Graspable", param) for param in parameters] + [
        ("On", param1, param2, DEFAULT_SHAPE)
        for param1, param2, in get_pairs(parameters)
    ]  # Graspable | Movable
    PROBLEMS.append(
        lambda args, num=num: Task(
            goal_parts=[
                Exists(parameters, And(*conditions)),
            ],
            arms=args.arms,
            skills=DEFAULT_SKILLS,
        )
    )
    PROBLEMS[-1].__name__ = f"stack_{num}"

##################################################
