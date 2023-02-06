from run_planner import create_parser, main
import pytest

def test_run_planner_problem0_pr2():
    parser = create_parser()
    args = parser.parse_args(["--robot", "pr2", "--goal", "all_green", "--world", "problem0", "--teleport", "--segmentation"])
    main(args)
