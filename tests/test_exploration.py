from run_planner import create_parser, main
import pytest

def test_run_planner_lamb():
    parser = create_parser()
    args = parser.parse_args(["--robot", "movo", "--goal", "all_green", "--world", "vanamo_m0m_chair", "--exploration", "--base-planner", "lamb"])
    main(args)

def test_run_planner_astar():
    parser = create_parser()
    args = parser.parse_args(["--robot", "movo", "--goal", "all_green", "--world", "vanamo_m0m", "--exploration", "--base-planner", "astar"])
    main(args)
