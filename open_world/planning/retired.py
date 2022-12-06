import copy
from itertools import product

from pddlstream.language.constants import (
    EXISTS,
    FORALL,
    IMPLY,
    QUANTIFIERS,
    Fact,
    get_args,
    get_prefix,
)
from pddlstream.language.conversion import list_from_conjunction

from open_world.planning.planner import replace_param, replace_params
from open_world.simulation.utils import partition

NECESSARY_PREDS = {
    # Necessary conditions (rules)
    "On": ["Stackable"],
    "In": ["Droppable"],
    "AtLeft": ["Stackable"],
    "AtRight": ["Stackable"],
}


def infer_affordances(task, init, objects):
    # NOTE(caelan): this is me trying to be cute and not critical
    # TODO: instead instantiate while omitting the fluent condition
    fn = lambda f: get_prefix(f) in NECESSARY_PREDS
    new_init = []
    for parent in task.goal_parts:
        prefix = get_prefix(parent)
        if prefix in NECESSARY_PREDS:
            new_init.extend(
                Fact(pred, get_args(parent)) for pred in NECESSARY_PREDS[prefix]
            )
        elif prefix in QUANTIFIERS:
            assert len(parent) == 3
            parameters = parent[1]
            if len(parameters) == 1:
                child = parent[2]
                if prefix == EXISTS:
                    fluents, statics = partition(fn, list_from_conjunction(child))
                elif prefix == FORALL:
                    assert get_prefix(child) == IMPLY
                    statics = list_from_conjunction(child[1])
                    fluents = list(filter(fn, list_from_conjunction(child[2])))
                else:
                    raise ValueError(prefix)
                for fluent in fluents:
                    necessary = [
                        Fact(pred, get_args(fluent))
                        for pred in NECESSARY_PREDS[get_prefix(fluent)]
                    ]
                    for obj in objects:
                        if all(
                            replace_param(static, obj) in init for static in statics
                        ):
                            new_init.extend(
                                replace_param(fact, obj) for fact in necessary
                            )
            elif len(parameters) > 1:
                # Todo: merge this with the previous case
                child = parent[2]
                if prefix == EXISTS:
                    fluents, statics = partition(fn, list_from_conjunction(child))
                for fluent in fluents:
                    necessary = [
                        Fact(pred, get_args(fluent))
                        for pred in NECESSARY_PREDS[get_prefix(fluent)]
                    ]
                    obj_products_args = [objects for _ in range(len(parameters))]
                    obj_products = product(*obj_products_args)
                    for obj_list in obj_products:
                        assert len(obj_list) == len(parameters)
                        # Check no duplicates
                        if len(set(obj_list)) == len(obj_list):
                            param_dict = {
                                param: obj_list[obj_index]
                                for obj_index, param in enumerate(parameters)
                            }
                            if all(
                                replace_params(static, param_dict) in init
                                for static in statics
                            ):  # TODO(curtisa): statics might not be defined
                                new_init.extend(
                                    replace_params(fact, param_dict)
                                    for fact in necessary
                                )

    return new_init


#######################################################


STATIC_PREDS = [
    "Movable",
    "Stackable",
    "Droppable",
    "CanPush",
    "CanPick",
    "Material",
    "CanContain",
    "Graspable",
]


def goal_factorize(task_parts, init):
    """
    This function splits the goal into a bunch of mini parts based on implicature and static predicates
    """
    new_task_parts = []
    for task_part in task_parts:
        if task_part[0] == FORALL:
            # variables = task_part[1]
            parts = task_part[2]
            if parts[0] == IMPLY:
                condition, goal_effect = parts[1:3]
                condition_pred, param = condition

                # Find actions which satisfy conditions
                for fact in init:
                    if fact[0] == condition_pred and condition_pred in STATIC_PREDS:
                        # Todo: generalizd across static types
                        assert len(condition) == 2
                        lst_goal_effect = list(copy.copy(goal_effect))
                        lst_goal_effect[1] = fact[1]

                        new_task_parts.append(tuple(lst_goal_effect))
        else:
            new_task_parts.append(task_part)
    return new_task_parts
