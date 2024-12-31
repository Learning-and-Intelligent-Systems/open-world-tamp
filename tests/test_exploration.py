import pytest

from run_planner import create_parser, main

# def test_run_planner_lamb():
#     parser = create_parser()
#     args = parser.parse_args(["--robot", "movo", "--goal", "all_green", "--world", "vanamo_m0m_chair", "--exploration", "--base-planner", "lamb"])
#     main(args)

# def test_run_planner_astar():
#     parser = create_parser()
#     args = parser.parse_args(["--robot", "movo", "--goal", "all_green", "--world", "vanamo_m0m", "--exploration", "--base-planner", "astar"])
#     main(args)


def test_run_planner_astar_movo():
    parser = create_parser()
    args = parser.parse_args(
        [
            "--robot",
            "movo",
            "--goal",
            "none",
            "--world",
            "empty_room",
            "--exploration",
            "--base-planner",
            "astar",
        ]
    )
    main(args)


def test_run_planner_astar_spot():
    parser = create_parser()
    args = parser.parse_args(
        [
            "--robot",
            "spot",
            "--goal",
            "none",
            "--world",
            "empty_room",
            "--exploration",
            "--base-planner",
            "astar",
        ]
    )
    main(args)
