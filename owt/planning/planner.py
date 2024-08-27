import math
from itertools import combinations, product

import numpy as np
from open_world.planning.pouring import get_plan_pour_fn
from open_world.planning.primitives import (Command, GroupConf, RelativePose,
                                            Sequence)
from open_world.planning.pushing import get_plan_push_fn
from open_world.planning.streams import (BASE_COST,
                                         get_cfree_pregrasp_pose_test,
                                         get_cfree_traj_pose_test,
                                         get_grasp_gen_fn,
                                         get_placement_gen_fn,
                                         get_plan_drop_fn, get_plan_inspect_fn,
                                         get_plan_motion_fn, get_plan_pick_fn,
                                         get_plan_place_fn, get_pose_cost_fn,
                                         get_reachability_test,
                                         get_test_cfree_pose_pose)
from open_world.simulation.control import simulate_controller
from open_world.simulation.entities import BOWL
from open_world.simulation.environment import BIN, set_gripper_friction
from open_world.simulation.utils import (find_closest_color,
                                         get_color_distance, sorted_union)
from pddlstream.algorithms.algorithm import reset_globals
from pddlstream.algorithms.meta import analyze_goal
from pddlstream.algorithms.serialized import solve_all_goals, solve_next_goal
from pddlstream.language.constants import (NOT, And, Equal, Exists, Fact,
                                           ForAll, Imply, Not, PDDLProblem,
                                           Solution, get_args, get_prefix,
                                           is_head, print_solution)
from pddlstream.language.conversion import replace_expression
from pddlstream.language.external import never_defer
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.language.stream import PartialInputs, StreamInfo
from pddlstream.utils import Profiler, get_file_path, lowercase, read
from pybullet_tools.utils import (CHROMATIC_COLORS, INF, YELLOW, LockRenderer,
                                  enable_gravity, flatten, get_difference_fn,
                                  is_center_on_aabb, wait_for_duration,
                                  wait_if_gui)

MOST = "most"  # superlatives, ordinals
LEAST = "least"

CONTAINERS = [BOWL, BIN]

COLORS = {
    "yellow": YELLOW,
}
COLORS.update(CHROMATIC_COLORS)


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
    conditions = [Not(Equal(o1, o2)) for o1, o2 in combinations(objects, r=2)]
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
        # 'sample-leftof': from_fn(get_cardinal_sample(robot, direction="leftof", **kwargs)),
        # 'sample-aheadof': from_fn(get_cardinal_sample(robot, direction="aheadof", **kwargs)),
        "sample-grasp": from_gen_fn(
            get_grasp_gen_fn(robot, [table], grasp_mode=grasp_mode, **kwargs)
        ),
        "sample-placement": from_gen_fn(
            get_placement_gen_fn(robot, [table], environment=obstacles, **kwargs)
        ),
        "plan-push": from_fn(get_plan_push_fn(robot, environment=obstacles, **kwargs)),
        "plan-pour": from_fn(get_plan_pour_fn(robot, environment=obstacles, **kwargs)),
        "plan-pick": from_fn(get_plan_pick_fn(robot, environment=obstacles, **kwargs)),
        # 'plan-mobile-pick': from_gen_fn(get_plan_mobile_pick_fn(robot, environment=obstacles, **kwargs)),
        "plan-place": from_fn(
            get_plan_place_fn(robot, environment=obstacles, **kwargs)
        ),
        # 'plan-mobile-place': from_gen_fn(get_plan_mobile_place_fn(robot, environment=obstacles, **kwargs)),
        "plan-motion": from_fn(
            get_plan_motion_fn(robot, environment=obstacles, **kwargs)
        ),
        "plan-drop": from_fn(get_plan_drop_fn(robot, environment=obstacles, **kwargs)),
        "plan-inspect": from_fn(
            get_plan_inspect_fn(robot, environment=obstacles, **kwargs)
        ),
        "PoseCost": get_pose_cost_fn(robot, **kwargs),
    }

    # if(mobile_base):
    #     stream_map.update(
    #         {"plan-mobile-pick": from_gen_fn(
    #             get_plan_mobile_pick_fn(robot, environment=obstacles, **kwargs)
    #          ),
    #          "plan-mobile-place": from_gen_fn(
    #             get_plan_mobile_place_fn(robot, environment=obstacles, **kwargs)
    #          )})
    # else:
    #     stream_map.update(
    #         {"plan-pick": from_fn(get_plan_pick_fn(robot, environment=obstacles, **kwargs)),
    #          "plan-place": from_fn(
    #              get_plan_place_fn(robot, environment=obstacles, **kwargs)
    #          )})

    # stream_map = DEBUG # TODO: run in debug mode first to reassign labels to objects if need be

    # stream_info = {name: StreamInfo(opt_gen_fn=PartialInputs(unique=True)) for name in stream_map}
    stream_info = {
        #'test-reachable': StreamInfo(),
        "test-cfree-pose-pose": StreamInfo(
            p_success=1e-3, eager=False, verbose=verbose
        ),
        "test-cfree-pregrasp-pose": StreamInfo(p_success=1e-2, verbose=verbose),
        "test-cfree-traj-pose": StreamInfo(p_success=1e-1, verbose=verbose),
        # 'sample-grasp': StreamInfo(),
        # 'sample-leftof': StreamInfo(overhead=1e1, opt_gen_fn=PartialInputs(unique=True)),
        # 'sample-aheadof': StreamInfo(overhead=1e1, opt_gen_fn=PartialInputs(unique=True)),
        "sample-placement": StreamInfo(
            overhead=1e1, opt_gen_fn=PartialInputs(unique=True)
        ),
        "plan-push": StreamInfo(overhead=1e1, eager=True),
        "plan-pour": StreamInfo(overhead=1e1),
        "plan-pick": StreamInfo(overhead=1e1),
        "plan-drop": StreamInfo(overhead=1e1),
        #'plan-mobile-pick': StreamInfo(),
        "plan-place": StreamInfo(overhead=1e1),
        #'plan-mobile-place': StreamInfo(),
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
    # obstacles = belief.obstacles # belief.surfaces
    arms = sorted(robot.manipulators)
    print("Num arms: " + str(arms))

    surfaces = belief.known_surfaces
    table = surfaces[0]
    regions = surfaces[1:]

    all_objects = sorted_union(
        belief.known_surfaces, objects
    )  # TODO: objects is kwargs
    containers = {obj for obj in all_objects if obj.category in CONTAINERS}
    fixed_objects = sorted_union(belief.known_surfaces, containers)
    # fixed_objects = belief.obstacles
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
    controllable.extend(robot.arm_from_side(side) for side in arms)
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
                # ('ClosestColor', find_closest_object(region, containers), region),
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
                # ('ClosestColor', find_closest_object(container, regions), container),
            ]
        )

    init.extend(get_attribute_facts(objects))  # all_objects | objects

    init.extend(("Controllable", group) for group in controllable)
    init_confs = {
        group: GroupConf(robot, group, important=True, **kwargs)
        for group in robot.groups
    }
    for group, conf in init_confs.items():
        # if any(ty in group for ty in ['gripper', 'head', 'torso']):
        #    continue
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
        # if 'arm' in group:
        #     init.append(('Arm', group))

    init_poses = {
        obj: RelativePose(obj, important=True, **kwargs) for obj in all_objects
    }
    for obj, pose in init_poses.items():
        init.extend(
            [
                # ('Object', obj), # TODO: confirm that this doesn't interfere
                (
                    "Category",
                    obj,
                    obj.category,
                ),  # TODO: distinguish between category and (YCB) instance
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
    for obj in (
        set(objects) - stowed_objects
    ):  # TODO: reordered so stowed_objects doesn't make sense
        if obj not in fixed_objects:
            init.extend(
                [
                    ("Movable", obj),
                    ("Graspable", obj),
                    ("Stackable", obj, table),
                    # ('CanPush', obj),
                    ("CanPick", obj),
                    # ('CanContain', obj) # TODO: unused
                ]
            )
            init.extend(
                ("Can{}".format(skill.capitalize()), obj) for skill in task.skills
            )

        if hasattr(obj, "contains"):  # TODO: otherwise error when observable
            for material in obj.contains:
                init.extend(
                    [
                        ("CanPour", obj),
                        ("Contains", obj, material),
                        ("Material", material),
                    ]
                )

    new_init = []
    # new_init.extend(infer_affordances(task, init, objects))
    new_init.extend(
        ("Stackable", obj, surface)
        for obj, surface in product(movable_objects, surfaces)
        if obj != surface
    )

    new_init.extend(
        ("Stackable", obj, surface)
        for obj, surface in product(movable_objects, movable_objects)
        if obj != surface
    )

    new_init.extend(
        ("Droppable", obj, container)
        for obj, container in product(movable_objects, containers)
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
            if is_center_on_aabb(
                obj,
                surface.get_shape_aabb(),
                above_epsilon=INF,
                below_epsilon=INF,
                **kwargs
            ):
                # is_placed_on_aabb | is_center_on_aabb
                # TODO: test stream
                init.append(
                    ("Supported", obj, init_poses[obj], surface, init_poses[surface])
                )
        elif pred == "Droppable":
            obj, container = get_args(fact)
            init.append(Equal(("DropCost", obj, container), BASE_COST))
            if is_center_on_aabb(
                obj,
                container.get_shape_aabb(),
                above_epsilon=INF,
                below_epsilon=INF,
                **kwargs
            ):
                # TODO: add obj as an obstacle
                stowed_objects.add(obj)
                init.append(("In", obj, container))
                # TODO: update belief with history
        else:
            raise NotImplementedError(fact)

    ##########

    rest_confs = {}
    for side in arms:
        # TODO: combine arm and gripper
        arm = robot.arm_from_side(side)

        q = (
            robot.default_mobile_base_arm
            if mobile_base
            else robot.default_fixed_base_arm
        )
        rest_conf = GroupConf(robot, arm, robot.arm_conf(arm, q), **kwargs)
        joints = rest_conf.joints
        difference = get_difference_fn(robot, joints, **kwargs)(
            rest_conf.positions, init_confs[arm].positions
        )
        if np.allclose(
            np.absolute(difference),
            np.zeros(len(difference)),
            rtol=0,
            atol=math.radians(2),
        ):
            rest_conf = init_confs[arm]
        rest_confs[arm] = rest_conf

        init.extend(
            [
                ("Arm", arm),
                ("ArmEmpty", arm),
                ("Conf", arm, rest_conf),
                ("RestConf", arm, rest_conf),
            ]
        )

    ##########

    task_parts = []
    task_parts.extend(task.goal_parts)
    # task_parts = goal_factorize(task_parts, init)

    goal_confs = [
        # init_confs['base'],
        # GroupConf(robot, 'base', [-2.5, -2.5, 0]),
        # GroupConf(robot, goal_arm, np.zeros(len(pr2.get_group_joints(goal_arm)))),
    ]
    if task.return_init:
        # TODO: allow both arms to move to their rest conf initially
        goal_confs.extend(
            rest_confs[group] if (group in rest_confs) else init_confs[group]
            for group in controllable
        )

    reset_parts = []
    if task.empty_arms:
        reset_parts.extend(("ArmEmpty", robot.arm_from_side(side)) for side in arms)
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

    # TODO: better function in utils
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


def plan_pddlstream(belief, task, debug=False, serialize=False, *args, **kwargs):
    reset_globals()
    problem, stream_info = create_pddlstream(belief, task, *args, **kwargs)
    if problem is None:
        return Solution(None, INF, [])

    profiler = Profiler()
    profiler.save()
    with LockRenderer(lock=not debug, **kwargs):
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
            success_cost=INF,
            max_time=INF,
            max_memory=INF,
            max_restarts=INF,
            iteration_time=1.5 * 60,
            unit_efforts=True,
            max_effort=INF,
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
        flatten(
            args[-1].commands for name, args in plan if isinstance(args[-1], Command)
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
            wait_if_gui()
        else:
            wait_for_duration(time_step)
    return state


def simulate_sequence(robot, sequence, hook=None, **kwargs):
    # TODO: estimate duration
    assert sequence is not None
    enable_gravity()
    set_gripper_friction(robot)
    simulate_controller(sequence.controller(**kwargs), hook=hook)
