from run_planner import create_parser, main
import pytest

def test_run_planner_problem0_pr2():
    parser = create_parser()
    args = parser.parse_args(["--robot", "movo", "--goal", "all_green", "--world", "vanamo_m0m_chair", "--exploration"])
    main(args)
