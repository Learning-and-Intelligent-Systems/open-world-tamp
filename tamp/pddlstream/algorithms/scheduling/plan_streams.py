from __future__ import print_function

import copy

from collections import defaultdict, namedtuple

from pddlstream.algorithms.downward import get_problem, task_from_domain_problem, get_cost_scale, \
    conditions_hold, apply_action, scale_cost, fd_from_fact, make_domain, make_predicate, evaluation_from_fd, \
    plan_preimage, fact_from_fd, USE_FORBID, pddl_from_instance, parse_action
from pddlstream.algorithms.instantiate_task import instantiate_task, sas_from_instantiated, FD_INSTANTIATE
from pddlstream.algorithms.scheduling.add_optimizers import add_optimizer_effects, \
    using_optimizers, recover_simultaneous
from pddlstream.algorithms.scheduling.apply_fluents import convert_fluent_streams
from pddlstream.algorithms.scheduling.negative import recover_negative_axioms, convert_negative
from pddlstream.algorithms.scheduling.postprocess import postprocess_stream_plan
from pddlstream.algorithms.scheduling.recover_axioms import recover_axioms_plans
from pddlstream.algorithms.scheduling.recover_functions import compute_function_plan
from pddlstream.algorithms.scheduling.recover_streams import get_achieving_streams, extract_stream_plan, \
    evaluations_from_stream_plan
from pddlstream.algorithms.scheduling.stream_action import add_stream_actions
from pddlstream.algorithms.scheduling.utils import partition_results, \
    add_unsatisfiable_to_goal, get_instance_facts
from pddlstream.algorithms.search import solve_from_task
from pddlstream.algorithms.advanced import UNIVERSAL_TO_CONDITIONAL
from pddlstream.language.constants import Not, get_prefix, EQ, FAILED, OptPlan, Action
from pddlstream.language.conversion import obj_from_pddl_plan, evaluation_from_fact, \
    fact_from_evaluation, transform_plan_args, transform_action_args, obj_from_pddl
from pddlstream.language.external import Result
from pddlstream.language.exogenous import get_fluent_domain
from pddlstream.language.function import Function
from pddlstream.language.stream import StreamResult
from pddlstream.language.optimizer import UNSATISFIABLE
from pddlstream.language.statistics import compute_plan_effort
from pddlstream.language.temporal import SimplifiedDomain, solve_tfd
from pddlstream.language.write_pddl import get_problem_pddl
from pddlstream.language.object import Object
from pddlstream.utils import Verbose, INF, topological_sort, get_ancestors

RENAME_ACTIONS = True
#RENAME_ACTIONS = not USE_FORBID

OptSolution = namedtuple('OptSolution', ['stream_plan', 'opt_plan', 'cost']) # TODO: move to the below
#OptSolution = namedtuple('OptSolution', ['stream_plan', 'action_plan', 'cost', 'supporting_facts', 'axiom_plan'])

##################################################

def add_stream_efforts(node_from_atom, instantiated, effort_weight, **kwargs):
    if effort_weight is None:
        return
    # TODO: make effort just a multiplier (or relative) to avoid worrying about the scale
    # TODO: regularize & normalize across the problem?
    #efforts = []
    for instance in instantiated.actions:
        # TODO: prune stream actions here?
        # TODO: round each effort individually to penalize multiple streams
        facts = get_instance_facts(instance, node_from_atom)
        #effort = COMBINE_OP([0] + [node_from_atom[fact].effort for fact in facts])
        stream_plan = []
        extract_stream_plan(node_from_atom, facts, stream_plan)
        effort = compute_plan_effort(stream_plan, **kwargs)
        instance.cost += scale_cost(effort_weight*effort)
        # TODO: store whether it uses shared/unique outputs and prune too expensive streams
        #efforts.append(effort)
    #print(min(efforts), efforts)

##################################################

def rename_instantiated_actions(instantiated, rename):
    # TODO: rename SAS instead?
    actions = instantiated.actions[:]
    renamed_actions = []
    action_from_name = {}
    for i, action in enumerate(actions):
        renamed_actions.append(copy.copy(action))
        renamed_name = 'a{}'.format(i) if rename else action.name
        renamed_actions[-1].name = '({})'.format(renamed_name)
        action_from_name[renamed_name] = action # Change reachable_action_params?
    instantiated.actions[:] = renamed_actions
    return action_from_name

##################################################

def get_plan_cost(action_plan, cost_from_action):
    if action_plan is None:
        return INF
    # TODO: return cost per action instance
    #return sum([0.] + [instance.cost for instance in action_plan])
    scaled_cost = sum([0.] + [cost_from_action[instance] for instance in action_plan])
    return scaled_cost / get_cost_scale()

def instantiate_optimizer_axioms(instantiated, domain, results):
    # Needed for instantiating axioms before adding stream action effects
    # Otherwise, FastDownward will prune these unreachable axioms
    # TODO: compute this first and then apply the eager actions
    stream_init = {fd_from_fact(result.stream_fact)
                   for result in results if isinstance(result, StreamResult)}
    evaluations = list(map(evaluation_from_fd, stream_init | instantiated.atoms))
    temp_domain = make_domain(predicates=[make_predicate(UNSATISFIABLE, [])],
                              axioms=[ax for ax in domain.axioms if ax.name == UNSATISFIABLE])
    temp_problem = get_problem(evaluations, Not((UNSATISFIABLE,)), temp_domain)
    # TODO: UNSATISFIABLE might be in atoms making the goal always infeasible
    with Verbose():
        # TODO: the FastDownward instantiation prunes static preconditions
        use_fd = False if using_optimizers(results) else FD_INSTANTIATE
        new_instantiated = instantiate_task(task_from_domain_problem(temp_domain, temp_problem),
                                            use_fd=use_fd, check_infeasible=False, prune_static=False)
        assert new_instantiated is not None
    instantiated.axioms.extend(new_instantiated.axioms)
    instantiated.atoms.update(new_instantiated.atoms)

##################################################

def recover_partial_orders(stream_plan, node_from_atom):
    # Useful to recover the correct DAG
    partial_orders = set()
    for child in stream_plan:
        # TODO: account for fluent objects
        for fact in child.get_domain():
            parent = node_from_atom[fact].result
            if parent is not None:
                partial_orders.add((parent, child))
    #stream_plan = topological_sort(stream_plan, partial_orders)
    return partial_orders

def recover_stream_plan(evaluations, current_plan, opt_evaluations, goal_expression, domain, node_from_atom,
                        action_plan, axiom_plans, negative, replan_step):
    # Universally quantified conditions are converted into negative axioms
    # Existentially quantified conditions are made additional preconditions
    # Universally quantified effects are instantiated by doing the cartesian produce of types (slow)
    # Added effects cancel out removed effects
    # TODO: node_from_atom is a subset of opt_evaluations (only missing functions)
    real_task = task_from_domain_problem(domain, get_problem(evaluations, goal_expression, domain))
    opt_task = task_from_domain_problem(domain, get_problem(opt_evaluations, goal_expression, domain))
    negative_from_name = {external.blocked_predicate: external for external in negative if external.is_negated}
    real_states, full_plan = recover_negative_axioms(
        real_task, opt_task, axiom_plans, action_plan, negative_from_name)
    function_plan = compute_function_plan(opt_evaluations, action_plan)

    full_preimage = plan_preimage(full_plan, []) # Does not contain the stream preimage!
    negative_preimage = set(filter(lambda a: a.predicate in negative_from_name, full_preimage))
    negative_plan = convert_negative(negative_preimage, negative_from_name, full_preimage, real_states)
    function_plan.update(negative_plan)
    # TODO: OrderedDict for these plans

    # TODO: this assumes that actions do not negate preimage goals
    positive_preimage = {l for l in (set(full_preimage) - real_states[0] - negative_preimage) if not l.negated}
    steps_from_fact = {fact_from_fd(l): full_preimage[l] for l in positive_preimage}
    last_from_fact = {fact: min(steps) for fact, steps in steps_from_fact.items() if get_prefix(fact) != EQ}
    #stream_plan = reschedule_stream_plan(evaluations, target_facts, domain, stream_results)
    # visualize_constraints(map(fact_from_fd, target_facts))

    for result, step in function_plan.items():
        for fact in result.get_domain():
            last_from_fact[fact] = min(step, last_from_fact.get(fact, INF))

    # TODO: get_steps_from_stream
    stream_plan = []
    last_from_stream = dict(function_plan)
    for result in current_plan: # + negative_plan?
        # TODO: actually compute when these are needed + dependencies
        last_from_stream[result] = 0
        if isinstance(result.external, Function) or (result.external in negative):
            if len(action_plan) > replan_step:
                raise NotImplementedError() # TODO: deferring negated optimizers
            # Prevents these results from being pruned
            function_plan[result] = replan_step
        else:
            stream_plan.append(result)

    curr_evaluations = evaluations_from_stream_plan(evaluations, stream_plan, max_effort=None)
    extraction_facts = set(last_from_fact) - set(map(fact_from_evaluation, curr_evaluations))
    extract_stream_plan(node_from_atom, extraction_facts, stream_plan)

    # Recomputing due to postprocess_stream_plan
    stream_plan = postprocess_stream_plan(evaluations, domain, stream_plan, last_from_fact)
    node_from_atom = get_achieving_streams(evaluations, stream_plan, max_effort=None)
    fact_sequence = [set(result.get_domain()) for result in stream_plan] + [extraction_facts]
    for facts in reversed(fact_sequence): # Bellman ford
        for fact in facts: # could flatten instead
            result = node_from_atom[fact].result
            if result is None:
                continue
            step = last_from_fact[fact] if result.is_deferrable() else 0
            last_from_stream[result] = min(step, last_from_stream.get(result, INF))
            for domain_fact in result.instance.get_domain():
                last_from_fact[domain_fact] = min(last_from_stream[result], last_from_fact.get(domain_fact, INF))
    stream_plan.extend(function_plan)

    partial_orders = recover_partial_orders(stream_plan, node_from_atom)
    bound_objects = set()
    for result in stream_plan:
        if (last_from_stream[result] == 0) or not result.is_deferrable(bound_objects=bound_objects):
            for ancestor in get_ancestors(result, partial_orders) | {result}:
                # TODO: this might change descendants of ancestor. Perform in a while loop.
                last_from_stream[ancestor] = 0
                if isinstance(ancestor, StreamResult):
                    bound_objects.update(out for out in ancestor.output_objects if out.is_unique())

    #local_plan = [] # TODO: not sure what this was for
    #for fact, step in sorted(last_from_fact.items(), key=lambda pair: pair[1]): # Earliest to latest
    #    print(step, fact)
    #    extract_stream_plan(node_from_atom, [fact], local_plan, last_from_fact, last_from_stream)

    # Each stream has an earliest evaluation time
    # When computing the latest, use 0 if something isn't deferred
    # Evaluate each stream as soon as possible
    # Option to defer streams after a point in time?
    # TODO: action costs for streams that encode uncertainty
    state = set(real_task.init)
    remaining_results = list(stream_plan)
    first_from_stream = {}
    #assert 1 <= replan_step # Plan could be empty
    for step, instance in enumerate(action_plan):
        for result in list(remaining_results):
            # TODO: could do this more efficiently if need be
            domain = result.get_domain() + get_fluent_domain(result)
            if conditions_hold(state, map(fd_from_fact, domain)):
                remaining_results.remove(result)
                certified = {fact for fact in result.get_certified() if get_prefix(fact) != EQ}
                state.update(map(fd_from_fact, certified))
                if step != 0:
                    first_from_stream[result] = step
        # TODO: assumes no fluent axiom domain conditions
        apply_action(state, instance)
    #assert not remaining_results # Not true if retrace
    if first_from_stream:
        replan_step = min(replan_step, *first_from_stream.values())

    eager_plan = []
    results_from_step = defaultdict(list)
    for result in stream_plan:
        earliest_step = first_from_stream.get(result, 0) # exogenous
        latest_step = last_from_stream.get(result, 0) # defer
        #assert earliest_step <= latest_step
        defer = replan_step <= latest_step
        if not defer:
            eager_plan.append(result)
        # We only perform a deferred evaluation if it has all deferred dependencies
        # TODO: make a flag that also allows dependencies to be deferred
        future = (earliest_step != 0) or defer
        if future:
            future_step = latest_step if defer else earliest_step
            results_from_step[future_step].append(result)

    # TODO: some sort of obj side-effect bug that requires obj_from_pddl to be applied last (likely due to fluent streams)
    eager_plan = convert_fluent_streams(eager_plan, real_states, action_plan, steps_from_fact, node_from_atom)
    combined_plan = []
    for step, action in enumerate(action_plan):
        combined_plan.extend(result.get_action() for result in results_from_step[step])
        combined_plan.append(transform_action_args(pddl_from_instance(action), obj_from_pddl))

    # TODO: the returned facts have the same side-effect bug as above
    # TODO: annotate when each preimage fact is used
    preimage_facts = {fact_from_fd(l) for l in full_preimage if (l.predicate != EQ) and not l.negated}
    for negative_result in negative_plan: # TODO: function_plan
        preimage_facts.update(negative_result.get_certified())
    for result in eager_plan:
        preimage_facts.update(result.get_domain())
        # Might not be able to regenerate facts involving the outputs of streams
        preimage_facts.update(result.get_certified()) # Some facts might not be in the preimage
    # TODO: record streams and axioms
    return eager_plan, OptPlan(combined_plan, preimage_facts)

##################################################

def solve_optimistic_temporal(domain, stream_domain, applied_results, all_results,
                              opt_evaluations, node_from_atom, goal_expression,
                              effort_weight, debug=False, **kwargs):
    # TODO: assert that the unused parameters are off
    assert domain is stream_domain
    #assert len(applied_results) == len(all_results)
    problem = get_problem(opt_evaluations, goal_expression, domain)
    with Verbose():
        instantiated = instantiate_task(task_from_domain_problem(domain, problem))
    if instantiated is None:
        return instantiated, None, None, INF
    problem = get_problem_pddl(opt_evaluations, goal_expression, domain.pddl)
    pddl_plan, makespan = solve_tfd(domain.pddl, problem, debug=debug, **kwargs)
    if pddl_plan is None:
        return instantiated, None, pddl_plan, makespan
    instance_from_action_args = defaultdict(list)
    for instance in instantiated.actions:
        name, args = parse_action(instance)
        instance_from_action_args[name, args].append(instance)
        #instance.action, instance.var_mapping
    action_instances = []
    for action in pddl_plan:
        instances = instance_from_action_args[action.name, action.args]
        if len(instances) != 1:
            for action in instances:
                action.dump()
        #assert len(instances) == 1 # TODO: support 2 <= case
        action_instances.append(instances[0])
    temporal_plan = obj_from_pddl_plan(pddl_plan) # pddl_plan is sequential
    return instantiated, action_instances, temporal_plan, makespan

def solve_optimistic_sequential(domain, stream_domain, applied_results, all_results,
                                opt_evaluations, node_from_atom, goal_expression,
                                effort_weight, debug=False, **kwargs):
    #print(sorted(map(fact_from_evaluation, opt_evaluations)))
    temporal_plan = None
    problem = get_problem(opt_evaluations, goal_expression, stream_domain)  # begin_metric
    with Verbose(verbose=debug):
        task = task_from_domain_problem(stream_domain, problem)
        instantiated = instantiate_task(task)
    if instantiated is None:
        return instantiated, None, temporal_plan, INF

    cost_from_action = {action: action.cost for action in instantiated.actions}
    add_stream_efforts(node_from_atom, instantiated, effort_weight)
    if using_optimizers(applied_results):
        add_optimizer_effects(instantiated, node_from_atom)
        # TODO: reachieve=False when using optimizers or should add applied facts
        instantiate_optimizer_axioms(instantiated, domain, all_results)
    action_from_name = rename_instantiated_actions(instantiated, RENAME_ACTIONS)
    # TODO: the action unsatisfiable conditions are pruned
    with Verbose(debug):
        sas_task = sas_from_instantiated(instantiated)
        #sas_task.metric = task.use_min_cost_metric
        sas_task.metric = True

    # TODO: apply renaming to hierarchy as well
    # solve_from_task | serialized_solve_from_task | abstrips_solve_from_task | abstrips_solve_from_task_sequential
    renamed_plan, _ = solve_from_task(sas_task, debug=debug, **kwargs)
    if renamed_plan is None:
        return instantiated, None, temporal_plan, INF

    action_instances = [action_from_name[name if RENAME_ACTIONS else '({} {})'.format(name, ' '.join(args))]
                        for name, args in renamed_plan]
    cost = get_plan_cost(action_instances, cost_from_action)
    return instantiated, action_instances, temporal_plan, cost

##################################################

def plan_streams(evaluations, goal_expression, domain, all_results, negative, effort_weight, max_effort,
                 simultaneous=False, reachieve=True, replan_actions=set(), **kwargs):
    # TODO: alternatively could translate with stream actions on real opt_state and just discard them
    # TODO: only consider axioms that have stream conditions?
    #reachieve = reachieve and not using_optimizers(all_results)
    #for i, result in enumerate(all_results):
    #    print(i, result, result.get_effort())
    applied_results, deferred_results = partition_results(
        evaluations, all_results, apply_now=lambda r: not (simultaneous or r.external.info.simultaneous))
    stream_domain, deferred_from_name = add_stream_actions(domain, deferred_results)

    if reachieve and not using_optimizers(all_results):
        achieved_results = {n.result for n in evaluations.values() if isinstance(n.result, Result)}
        init_evaluations = {e for e, n in evaluations.items() if n.result not in achieved_results}
        applied_results = achieved_results | set(applied_results)
        evaluations = init_evaluations # For clarity

    # TODO: could iteratively increase max_effort
    node_from_atom = get_achieving_streams(evaluations, applied_results, # TODO: apply to all_results?
                                           max_effort=max_effort)
    opt_evaluations = {evaluation_from_fact(f): n.result for f, n in node_from_atom.items()}
    if UNIVERSAL_TO_CONDITIONAL or using_optimizers(all_results):
        goal_expression = add_unsatisfiable_to_goal(stream_domain, goal_expression)

    temporal = isinstance(stream_domain, SimplifiedDomain)
    optimistic_fn = solve_optimistic_temporal if temporal else solve_optimistic_sequential
    instantiated, action_instances, temporal_plan, cost = optimistic_fn(
        domain, stream_domain, applied_results, all_results, opt_evaluations,
        node_from_atom, goal_expression, effort_weight, **kwargs)
    if action_instances is None:
        return OptSolution(FAILED, FAILED, cost)

    action_instances, axiom_plans = recover_axioms_plans(instantiated, action_instances)
    # TODO: extract out the minimum set of conditional effects that are actually required
    #simplify_conditional_effects(instantiated.task, action_instances)
    stream_plan, action_instances = recover_simultaneous(
        applied_results, negative, deferred_from_name, action_instances)

    action_plan = transform_plan_args(map(pddl_from_instance, action_instances), obj_from_pddl)
    replan_step = min([step+1 for step, action in enumerate(action_plan)
                       if action.name in replan_actions] or [len(action_plan)+1]) # step after action application

    stream_plan, opt_plan = recover_stream_plan(evaluations, stream_plan, opt_evaluations, goal_expression, stream_domain,
        node_from_atom, action_instances, axiom_plans, negative, replan_step)
    if temporal_plan is not None:
        # TODO: handle deferred streams
        assert all(isinstance(action, Action) for action in opt_plan.action_plan)
        opt_plan.action_plan[:] = temporal_plan
    return OptSolution(stream_plan, opt_plan, cost)
