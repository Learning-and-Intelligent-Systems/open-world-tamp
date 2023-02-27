from run_planner import create_parser, main
import pytest

def test_run_planner_problem0_pr2():
    parser = create_parser()
    args = parser.parse_args(["--robot", "pr2", "--goal", "none", "--world", "problem0", "--teleport"])
    main(args)

def test_run_planner_problem0_panda():
    parser = create_parser()
    args = parser.parse_args(["--robot", "panda", "--goal", "none", "--world", "problem0", "--teleport"])
    main(args)

def test_run_planner_problem0_movo():
    parser = create_parser()
    args = parser.parse_args(["--robot", "movo", "--goal", "none", "--world", "problem0", "--teleport"])
    main(args)