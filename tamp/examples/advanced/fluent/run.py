#!/usr/bin/env python

from __future__ import print_function

from pddlstream.algorithms.meta import solve, create_parser, FOCUSED_ALGORITHMS
from pddlstream.language.generator import from_test, from_fn, universe_test
from pddlstream.language.stream import StreamInfo
from pddlstream.language.constants import And, print_solution, PDDLProblem, Output
from pddlstream.utils import read, get_file_path

TRAJ = [0, 1]

def feasibility_test(o, fluents=set()):
    for fact in fluents:
        if fact[0] == 'ontable':
            o2, = fact[1:]
            if (o != o2) and (o2 == 'b2'):
                return False
    return True

def feasibility_fn(o, fluents=set()):
    if not feasibility_test(o, fluents=fluents):
        return None
    return Output(TRAJ)

def get_pddlstream_problem():
    # TODO: bug where a trajectory sample could be used in a different state than anticipated (don't return the sample)
    # TODO: enforce positive axiom preconditions requiring the state to be exactly some given value
    #       then, the can outputs can be used in other streams only present at that state
    # TODO: explicitly don't let the outputs of one fluent stream be the input to another on a different state

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    constant_map = {}
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    stream_map = {
        'sample-pickable': from_fn(feasibility_fn),
        'test-cleanable': from_test(universe_test), # TODO: bug because this stream is never called
        #'test-cleanable': from_fn(lambda o, fluents=set(): None if fluents else (TRAJ,)),
    }

    # Currently tests whether one can clean twice
    init = [
        ('Block', 'b1'),
        ('Block', 'b2'),
        ('OnTable', 'b1'),
        ('OnTable', 'b2'),
    ]

    #goal = ('Holding', 'b1')
    goal = And(
        ('Clean', 'b1'),
        ('Cooked', 'b1'),
    )

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

##################################################

def main():
    parser = create_parser()
    args = parser.parse_args()
    print('Arguments:', args)

    # TODO: maybe load problems as a domain explicitly
    problem = get_pddlstream_problem()
    #if args.algorithm not in FOCUSED_ALGORITHMS:
    #    raise RuntimeError('The {} algorithm does not support fluent streams'.format(args.algorithm))
    solution = solve(problem, algorithm=args.algorithm, unit_costs=args.unit)
    print_solution(solution)

if __name__ == '__main__':
    main()
