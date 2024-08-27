import pytest

from run_planner import create_parser, main


def test_run_planner_problem0_pr2():
    parser = create_parser()
    args = parser.parse_args(
        ["--robot", "pr2", "--goal", "all_green", "--world", "problem0", "--teleport"]
    )
    main(args)


def test_run_planner_problem0_panda():
    parser = create_parser()
    args = parser.parse_args(
        ["--robot", "panda", "--goal", "all_green", "--world", "problem0", "--teleport"]
    )
    main(args)


def test_run_planner_problem0_movo():
    parser = create_parser()
    args = parser.parse_args(
        ["--robot", "movo", "--goal", "all_green", "--world", "problem0", "--teleport"]
    )
    main(args)


def test_run_planner_problem0_spot():
    parser = create_parser()
    args = parser.parse_args(
        ["--robot", "spot", "--goal", "all_green", "--world", "problem0", "--teleport"]
    )
    main(args)
