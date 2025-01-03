import itertools
import math

import numpy as np
from pddlstream.algorithms.algorithm import reset_globals
from pddlstream.algorithms.meta import analyze_goal
from pddlstream.algorithms.serialized import solve_all_goals, solve_next_goal
from pddlstream.language.constants import (NOT, And, Equal, Exists, Fact,
                                           ForAll, Imply, Not, PDDLProblem,
                                           Solution, get_args, get_prefix,
                                           is_head, print_solution)
from pddlstream.language.conversion import replace_expression
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.language.stream import PartialInputs, StreamInfo
from pddlstream.utils import Profiler, get_file_path, lowercase, read

import owt.pb_utils as pbu
from owt.estimation.belief import Belief
from owt.planning.primitives import Command, GroupConf, RelativePose, Sequence
from owt.planning.pushing import get_plan_push_fn
from owt.planning.streams import (BASE_COST, get_cfree_pregrasp_pose_test,
                                  get_cfree_traj_pose_test, get_grasp_gen_fn,
                                  get_placement_gen_fn, get_plan_drop_fn,
                                  get_plan_motion_fn, get_plan_pick_fn,
                                  get_plan_place_fn, get_pose_cost_fn,
                                  get_reachability_test,
                                  get_test_cfree_pose_pose)
from owt.simulation.control import simulate_controller
from owt.simulation.entities import BOWL
from owt.simulation.environment import BIN, set_gripper_friction
from owt.simulation.utils import (find_closest_color, get_color_distance,
                                  sorted_union)

MOST = "most"  # superlatives, ordinals
LEAST = "least"

CONTAINERS = [BOWL, BIN]

COLORS = {
    "yellow": pbu.YELLOW,
}
COLORS.update(pbu.CHROMATIC_COLORS)


#######################################################


def get_object_color(obj):
    # color = get_color(obj)
    color = obj.color
    if color is None:
        return None
    return color


def get_color_from_obj(objects):
    color_from_body = {}
    for obj in objects:
        color = get_object_color(obj)
        if color is not None:
            color_from_body[obj] = color
    return color_from_body


def closest_color(target_color, color_from_body, **kwargs):
    if (target_color is None) or (not color_from_body):
        return None
    return min(
        color_from_body,
        key=lambda b: get_color_distance(color_from_body[b], target_color, **kwargs),
    )


def find_closest_object(target_object, candidate_objects, **kwargs):
    return closest_color(
        get_object_color(target_object), get_color_from_obj(candidate_objects), **kwargs
    )


def get_attribute_facts(objects, color_from_name=COLORS, hue_only=False, **kwargs):
    init = []
    color_from_body = get_color_from_obj(objects)
    for obj, color in color_from_body.items():
        color_name = find_closest_color(
            color, color_from_name=color_from_name, hue_only=hue_only, **kwargs
        )
        init.append(("Color", obj, color_name))

    if not color_from_body:
        return init
    for target_name, target_color in color_from_name.items():
        closest_body = closest_color(
            target_color, color_from_body, hue_only=hue_only, **kwargs
        )
        init.append(("ClosestColor", closest_body, target_name))

    return init


#######################################################

PARAM = "?o"


def replace_param(condition, value):
    fn = lambda arg: value if isinstance(arg, str) and (arg == PARAM) else arg
    if is_head(condition):
        # TODO: replace_expression converts to lowercase predicates
        return Fact(get_prefix(condition), map(fn, get_args(condition)))
    return replace_expression(condition, fn)


def replace_params(condition, values_dict):
    fn = lambda arg: (
        values_dict[arg]
        if isinstance(arg, str) and arg in list(values_dict.keys())
        else arg
    )
    if is_head(condition):
        # TODO: replace_expression converts to lowercase predicates
        return Fact(get_prefix(condition), map(fn, get_args(condition)))
    return replace_expression(condition, fn)


def ExistTuple(condition, num=1):
    objects = ["{}{}".format(PARAM, i + 1) for i in range(num)]
    conditions = [Not(Equal(o1, o2)) for o1, o2 in itertools.combinations(objects, r=2)]
    conditions.extend(replace_param(condition, obj) for obj in objects)
    return Exists(objects, And(*conditions))


def ExistsObject(static, fluent):
    return Exists([PARAM], And(static, fluent))


def ForAllObjects(static, fluent):
    return ForAll([PARAM], Imply(static, fluent))


########################


def HoldingCategory(category, arm):
    return ExistsObject(("Category", PARAM, category), ("ArmHolding", arm, PARAM))


def On(surface):
    return ("On", PARAM, surface)


def Contains(material):
    return ("Contains", PARAM, material)


def ObjectLeft(surface):
    return ExistsObject(("Movable", PARAM), ("AtLeft", PARAM, surface))


def ObjectRight(surface):
    return ExistsObject(("Movable", PARAM), ("AtRight", PARAM, surface))


def ObjectOn(surface, **kwargs):
    # TODO: existential_quantification in SS-Replan
    return ExistsObject(("Movable", PARAM), On(surface, **kwargs))


def ColorOn(color, surface, **kwargs):
    return ExistsObject(("Color", PARAM, color), On(surface, **kwargs))


def CategoryOn(category, surface, **kwargs):
    return ExistsObject(("Category", PARAM, category), On(surface, **kwargs))


def AllCategoryOn(category, surface, **kwargs):
    # TODO: could express in a negated form
    return ForAllObjects(("Category", PARAM, category), On(surface, **kwargs))


def AllObjectsOn(surface, **kwargs):
    return ForAllObjects(("Movable", PARAM), On(surface, **kwargs))


def AllCategoryContains(category, material, **kwargs):
    # TODO: could express in a negated form
    return ForAllObjects(("Category", PARAM, category), Contains(material, **kwargs))


#######################################################


def create_streams(
    belief, obstacles=[], mobile_base=False, grasp_mode="mesh", verbose=False, **kwargs
):
    robot = belief.robot
    # obstacles = belief.obstacles
    table = belief.known_surfaces[0]

    # TODO: pass the belief
    stream_map = {
        "test-reachable": from_test(get_reachability_test(robot, **kwargs)),
        "test-cfree-pose-pose": from_test(get_test_cfree_pose_pose(**kwargs)),
        "test-cfree-pregrasp-pose": from_test(get_cfree_pregrasp_pose_test(robot)),
        "test-cfree-traj-pose": from_test(get_cfree_traj_pose_test(robot)),
        "sample-grasp": from_gen_fn(
            get_grasp_gen_fn(robot, [table], grasp_mode=grasp_mode, **kwargs)
        ),
        "sample-placement": from_gen_fn(
            get_placement_gen_fn(robot, [table], environment=obstacles, **kwargs)
        ),
        "plan-push": from_fn(get_plan_push_fn(robot, environment=obstacles, **kwargs)),
        "plan-pick": from_fn(get_plan_pick_fn(robot, environment=obstacles, **kwargs)),
        "plan-place": from_fn(
            get_plan_place_fn(robot, environment=obstacles, **kwargs)
        ),
        "plan-motion": from_fn(
            get_plan_motion_fn(robot, environment=obstacles, **kwargs)
        ),
        "plan-drop": from_fn(get_plan_drop_fn(robot, environment=obstacles, **kwargs)),
        "PoseCost": get_pose_cost_fn(robot, **kwargs),
    }

    stream_info = {
        "test-cfree-pose-pose": StreamInfo(
            p_success=1e-3, eager=False, verbose=verbose
        ),
        "test-cfree-pregrasp-pose": StreamInfo(p_success=1e-2, verbose=verbose),
        "test-cfree-traj-pose": StreamInfo(p_success=1e-1, verbose=verbose),
        "sample-placement": StreamInfo(
            overhead=1e1, opt_gen_fn=PartialInputs(unique=True)
        ),
        "plan-push": StreamInfo(overhead=1e1, eager=True),
        "plan-pick": StreamInfo(overhead=1e1),
        "plan-drop": StreamInfo(overhead=1e1),
        "plan-place": StreamInfo(overhead=1e1),
        "plan-inspect": StreamInfo(overhead=1e1),
        "plan-motion": StreamInfo(overhead=1e2),
        "PoseCost": FunctionInfo(opt_fn=lambda *args: BASE_COST, eager=True),
    }
    return stream_map, stream_info


#######################################################


def create_pddlstream(
    belief,
    task,
    objects=[],
    mobile_base=False,
    stack=False,
    restrict_init=True,
    **kwargs
):
    robot = belief.robot
    manipulator_groups = sorted(robot.manipulators)
    print("Num arms: " + str(manipulator_groups))

    surfaces = belief.known_surfaces
    table = surfaces[0]
    regions = surfaces[1:]

    all_objects = sorted_union(belief.known_surfaces, objects)

    containers = {obj for obj in all_objects if obj.category in CONTAINERS}
    fixed_objects = sorted_union(belief.known_surfaces, containers)
    movable_objects = set(all_objects) - set(fixed_objects)
    if stack:
        surfaces = sorted_union(surfaces, objects)
    print("All:", all_objects)
    print("Containers:", containers)
    print("Fixed:", fixed_objects)
    print("Movable:", movable_objects)
    print("Surfaces:", surfaces)
    print("Regions:", regions)

    controllable = []
    controllable.extend(manipulator_groups)
    if mobile_base:
        controllable.append("base")

    init = [
        ("StillActing",),
        ("ConfidentInState",),
        Equal(("PushCost",), BASE_COST + 10),
    ]
    init.extend(task.init)

    for obj in movable_objects:
        init.extend(
            [
                ("ClosestColor", find_closest_object(obj, regions), obj),
                ("ClosestColor", find_closest_object(obj, containers), obj),
            ]
        )
    for region in regions:
        color = get_object_color(region)
        color_name = find_closest_color(color, hue_only=True)
        init.extend(
            [
                ("ClosestColor", find_closest_object(region, movable_objects), region),
                ("Region", region),
                ("Color", region, color_name),
            ]
        )
    for container in containers:
        init.extend(
            [
                (
                    "ClosestColor",
                    find_closest_object(container, movable_objects),
                    container,
                ),
            ]
        )

    init.extend(get_attribute_facts(objects))  # all_objects | objects

    init.extend(("Controllable", group) for group in controllable)
    init_confs = {
        group: GroupConf(robot, group, important=True, **kwargs)
        for group in robot.groups
    }
    for group, conf in init_confs.items():
        if group == "body":
            group = "base"

        init.extend(
            [
                ("Conf", group, conf),
                ("InitConf", group, conf),
                ("AtConf", group, conf),
                ("CanMove", group),
                Equal(("MoveCost", group), 1),
            ]
        )

    init_poses = {
        obj: RelativePose(obj, important=True, **kwargs) for obj in all_objects
    }
    for obj, pose in init_poses.items():
        init.extend(
            [
                (
                    "Category",
                    obj,
                    obj.category,
                ),
                ("Localized", obj),
                ("Pose", obj, pose),
                ("AtPose", obj, pose),
                ("InitPose", obj, pose),
                ("ConfidentInPose", obj, pose),
            ]
        )
        init.extend(
            Fact(predicate.capitalize(), (obj,) + tuple(args))
            for predicate, *args in obj.properties
        )

    stowed_objects = set()
    for obj in set(objects) - stowed_objects:
        if obj not in fixed_objects:
            init.extend(
                [
                    ("Movable", obj),
                    ("Graspable", obj),
                    ("Stackable", obj, table),
                    ("CanPick", obj),
                ]
            )
            init.extend(
                ("Can{}".format(skill.capitalize()), obj) for skill in task.skills
            )

        if hasattr(obj, "contains"):
            for material in obj.contains:
                init.extend(
                    [
                        ("CanPour", obj),
                        ("Contains", obj, material),
                        ("Material", material),
                    ]
                )

    new_init = []
    new_init.extend(
        ("Stackable", obj, surface)
        for obj, surface in itertools.product(movable_objects, surfaces)
        if obj != surface
    )

    new_init.extend(
        ("Stackable", obj, surface)
        for obj, surface in itertools.product(movable_objects, movable_objects)
        if obj != surface
    )

    new_init.extend(
        ("Droppable", obj, container)
        for obj, container in itertools.product(movable_objects, containers)
        if obj != container
    )

    init.extend(new_init)
    for fact in set(new_init):
        pred = get_prefix(fact)
        if pred == "Stackable":
            obj, surface = get_args(fact)
            place_cost = BASE_COST
            if surface != table:
                pass
            elif surface in belief.known_surfaces:
                place_cost += 2
            else:
                place_cost += 4
            init.append(Equal(("PlaceCost", obj, surface), place_cost))
            if pbu.is_center_on_aabb(
                obj,
                surface.get_shape_aabb(),
                above_epsilon=np.inf,
                below_epsilon=np.inf,
                **kwargs
            ):
                init.append(
                    ("Supported", obj, init_poses[obj], surface, init_poses[surface])
                )
        elif pred == "Droppable":
            obj, container = get_args(fact)
            init.append(Equal(("DropCost", obj, container), BASE_COST))
            if pbu.is_center_on_aabb(
                obj,
                container.get_shape_aabb(),
                above_epsilon=np.inf,
                below_epsilon=np.inf,
                **kwargs
            ):
                stowed_objects.add(obj)
                init.append(("In", obj, container))
        else:
            raise NotImplementedError(fact)

    ##########

    rest_confs = {}
    for group in manipulator_groups:
        q = robot.get_default_conf()[group]
        rest_conf = GroupConf(robot, group, q, **kwargs)
        joints = rest_conf.joints
        difference = pbu.get_difference_fn(robot, joints, **kwargs)(
            rest_conf.positions, init_confs[group].positions
        )
        if np.allclose(
            np.absolute(difference),
            np.zeros(len(difference)),
            rtol=0,
            atol=math.radians(2),
        ):
            rest_conf = init_confs[group]
        rest_confs[group] = rest_conf

        init.extend(
            [
                ("Arm", group),
                ("ArmEmpty", group),
                ("Conf", group, rest_conf),
                ("RestConf", group, rest_conf),
            ]
        )

    ##########

    task_parts = []
    task_parts.extend(task.goal_parts)
    goal_confs = []

    if task.return_init:
        goal_confs.extend(
            rest_confs[group] if (group in rest_confs) else init_confs[group]
            for group in controllable
        )

    reset_parts = []
    if task.empty_arms:
        reset_parts.extend(("ArmEmpty", group) for group in manipulator_groups)
    for conf in goal_confs:
        group = conf.group
        init.extend(
            [
                ("Conf", group, conf),
            ]
        )
        reset_parts.extend(
            [
                ("AtConf", group, conf),
            ]
        )

    goal_parts = reset_parts + task_parts
    print("Init:", sorted(init, key=lambda f: f[0]))
    print("Goal:", goal_parts)

    ##########

    domain_pddl = read(get_file_path(__file__, "pddl/domain.pddl"))
    constant_map = {"@{}".format(group): group for group in ["base", "head", "torso"]}
    stream_pddl = read(get_file_path(__file__, "pddl/stream.pddl"))
    stream_map, stream_info = create_streams(belief, obstacles=fixed_objects, **kwargs)

    ##########
    problem = PDDLProblem(
        domain_pddl, constant_map, stream_pddl, stream_map, init, And(*goal_parts)
    )
    if restrict_init:
        (
            domain_pddl,
            constant_map,
            stream_pddl,
            stream_map,
            init,
            _,
        ) = restrict_stackable(problem, belief.known_surfaces[:1])
    problem = PDDLProblem(
        domain_pddl, constant_map, stream_pddl, stream_map, init, goal_parts
    )

    return problem, stream_info


#######################################################


def restrict_stackable(problem, surfaces):
    goal_facts = analyze_goal(problem)

    if goal_facts is None:
        return None
    table = surfaces[0]
    restricted_predicates = lowercase("Stackable")

    goal_init = []
    for fact in goal_facts:
        predicate = get_prefix(fact)
        if (predicate != NOT) and (predicate.lower() in restricted_predicates):
            goal_init.append(fact)

    domain_pddl, constant_map, stream_pddl, stream_map, init, goal = problem
    nonstack_init = [
        fact
        for fact in init
        if (get_prefix(fact).lower() not in restricted_predicates)
        or (get_args(fact)[1] == table)
    ]
    init = nonstack_init + goal_init
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


#######################################################


def plan_pddlstream(
    belief: Belief, task, debug=False, serialize=False, *args, **kwargs
):
    reset_globals()
    problem, stream_info = create_pddlstream(belief, task, *args, **kwargs)
    if problem is None:
        return Solution(None, np.inf, [])

    profiler = Profiler()
    profiler.save()
    with pbu.LockRenderer(lock=not debug, **kwargs):
        solve_fn = (
            solve_next_goal if serialize else solve_all_goals
        )  # solve_first_goal | solve_next_goal
        solution = solve_fn(
            problem,
            stream_info=stream_info,
            replan_actions={"perceive"},
            initial_complexity=0,
            planner="ff-astar2",
            max_planner_time=10,
            unit_costs=False,
            success_cost=np.inf,
            max_time=np.inf,
            max_memory=np.inf,
            max_restarts=np.inf,
            iteration_time=1.5 * 60,
            unit_efforts=True,
            max_effort=np.inf,
            effort_weight=1,
            search_sample_ratio=5,
            verbose=True,
            debug=False,
        )
    belief.robot.remove_components()
    profiler.restore()
    print_solution(solution)
    return solution


def post_process(plan):
    if plan is None:
        return None

    sequence = Sequence(
        itertools.chain(
            *[args[-1].commands for name, args in plan if isinstance(args[-1], Command)]
        )
    )
    sequence.dump()
    return sequence


#######################################################
def iterate_sequence(
    state, sequence, time_step=5e-3, teleport=False, **kwargs
):  # None | INF
    assert sequence is not None
    for i, _ in enumerate(sequence.iterate(state, teleport=teleport, **kwargs)):
        state.propagate()
        if time_step is None:
            pbu.wait_if_gui()
        else:
            pbu.wait_for_duration(time_step)
    return state


def simulate_sequence(robot, sequence, hook=None, **kwargs):
    # TODO: estimate duration
    assert sequence is not None
    pbu.enable_gravity()
    set_gripper_friction(robot)
    simulate_controller(sequence.controller(**kwargs), hook=hook)
