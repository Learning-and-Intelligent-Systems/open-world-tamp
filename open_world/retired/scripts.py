import os
import time

import numpy as np
import pybullet as p
from pybullet_tools.utils import (
    INF,
    NULL_ID,
    WHITE,
    RayResult,
    add_button,
    add_data_path,
    add_parameter,
    add_pose_constraint,
    apply_alpha,
    control_joint,
    control_joints,
    create_obj,
    dump_body,
    get_joint_name,
    get_max_limit,
    get_max_limits,
    get_min_limits,
    get_mouse_events,
    get_movable_joints,
    get_pose,
    get_ray_from_to,
    invert,
    load_pybullet,
    multiply,
    parent_link_from_joint,
    read_button,
    read_counter,
    read_parameter,
    remove_body,
    remove_constraint,
    set_all_color,
    set_joint_positions,
    set_point,
    set_pose,
    set_preview,
    step_simulation,
    unit_point,
    wait_for_user,
    wait_if_gui,
)

from open_world.planning.grasping import close_gripper, get_grasp, get_pregrasp
from open_world.planning.primitives import RelativePose
from open_world.planning.streams import get_placement_gen_fn
from open_world.simulation.control import (
    interpolate_pose_controller,
    simulate_controller,
    stall_for_duration,
)
from open_world.simulation.entities import invert_dict
from open_world.simulation.utils import random_color

ATLASNET_DIR = "shape_completion/completed_models"


def test_atlasnet(scale=1e-1):
    # TODO: scale predictions
    # TODO: resample to obtain a mesh
    for file_name in sorted(os.listdir(ATLASNET_DIR)):
        name = file_name.split(".")[0]
        file_path = os.path.abspath(os.path.join(ATLASNET_DIR, file_name))
        obj = create_obj(file_path, color=WHITE)  # , **kwargs)
        wait_if_gui()
        remove_body(obj)


def test_pick(
    gripper,
    obj,
    grasp_tool,
    gripper_from_tool,
    gripper_joints,
    max_velocities,
    **kwargs
):
    closed_conf = np.zeros(len(gripper_joints))
    grasp = get_grasp(grasp_tool, gripper_from_tool)
    pregrasp = get_pregrasp(grasp_tool, gripper_from_tool, **kwargs)

    obj_pose = get_pose(obj)
    grasp_pose = multiply(obj_pose, invert(grasp))
    pregrasp_pose = multiply(obj_pose, invert(pregrasp))

    set_point(gripper, [0.5, 0.5, 1.5])
    initial_pose = get_pose(gripper)

    # for target_pose in interpolate_poses(initial_pose, pregrasp_pose,
    #                                      #pos_step_size=INF, ori_step_size=INF):
    #                                      pos_step_size=0.5, ori_step_size=np.pi / 4):
    #     handles = draw_pose(target_pose)
    #     step_controller(pose_controller(gripper, target_pose))
    #     remove_handles(handles)
    set_pose(gripper, pregrasp_pose)

    # add_pose_constraint(gripper, pose=grasp_pose) #, max_force=500.)
    simulate_controller(interpolate_pose_controller(gripper, grasp_pose))

    # TODO: no velocity constraint unless last waypoint
    pose_constraint = add_pose_constraint(gripper)
    # rigid_constraint = add_fixed_constraint(obj, gripper)
    simulate_controller(
        close_gripper(
            gripper, gripper_joints, closed_conf, obj, max_velocities=max_velocities
        )
    )
    # control_joints(gripper, gripper_joints) # TODO: don't change the gripper set point

    remove_constraint(pose_constraint)
    simulate_controller(interpolate_pose_controller(gripper, pregrasp_pose))

    # Tests drop
    # pose_constraint = add_pose_constraint(gripper)
    # joint_controller_hold(gripper, gripper_joints, open_conf)

    # rigid_constraint = add_fixed_constraint(obj, gripper)
    # More closely spaced slows the controller down
    simulate_controller(interpolate_pose_controller(gripper, initial_pose))


def add_gripper_gears(robot, gripper_group, max_force=1e5):
    gripper_joints = robot.get_group_joints(gripper_group)
    constraints = []
    for child_joint in robot.get_finger_links(gripper_joints):
        # TODO: create a constraint to keep the fingers centered
        # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py#L46
        parent_joint = parent_link_from_joint(robot, child_joint)
        c = p.createConstraint(
            robot,
            parent_joint,
            robot,
            child_joint,
            jointType=p.JOINT_GEAR,
            # jointAxis=[0, 0, 0],
            jointAxis=[1, 0, 0],
            parentFramePosition=unit_point(),
            childFramePosition=unit_point(),
        )
        ratio = get_max_limit(robot, parent_joint) / get_max_limit(robot, child_joint)
        p.changeConstraint(c, gearRatio=-ratio, maxForce=max_force)
        constraints.append(c)
    return constraints


def test_gears(robot):
    # TODO: friction_anchors
    # https://github.com/bulletphysics/bullet3/issues?q=is%3Aissue+grasp
    # https://github.com/bulletphysics/bullet3/issues/1936
    # https://github.com/bulletphysics/bullet3/search?l=Python&q=grasp
    # https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_robots/panda
    # https://www.youtube.com/watch?v=yKShjSTayco
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_data/sphere_small.urdf
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_data/lego/lego.urdf
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_robots/panda/loadpanda_grasp.py
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_kuka_diverse_object_grasping.py
    # diff ss-pybullet/models/franka_description/robots/panda_arm_hand.urdf ../external/pybullet/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf

    # TODO: disable joint hold on the gripper joints at the start
    gripper_group = "right_gripper"
    gripper_joints = robot.get_group_joints(gripper_group)
    set_joint_positions(
        robot, gripper_joints, get_min_limits(robot, gripper_joints)
    )  # get_max_limits

    # parent_joint = joint_from_name(robot, '{}_gripper_joint'.format(gripper_group[0]))
    # #parent_joint = gripper_joints[3]
    # joints = [parent_joint]
    # open_conf = get_max_limits(robot, joints)
    # closed_conf = get_min_limits(robot, joints)
    # print(get_joint_positions(robot, joints))
    # wait_if_gui()
    # set_joint_positions(robot, joints, np.array(open_conf))
    # wait_if_gui()

    # TODO: control controller to move back within the joints limits
    # TODO: replace PR2's grippers with Franka Panda FE grippers (helps for GraspNet as well
    # TODO: manually convert the PR2's gripper into parallel jaw gripper by controlling the finger tip

    # TODO: follow the following PyBullet control recommendation
    # https://github.com/bulletphysics/bullet3/blob/ddc47f932888a6ea3b4e11bd5ce73e8deba0c9a1/examples/pybullet/examples/mimicJointConstraint.py

    for joint in get_movable_joints(robot):
        if joint not in gripper_joints:
            control_joint(robot, joint)

    # control_joints(robot, gripper_joints) #[2:])
    # for joint in gripper_joints: #[2:]:
    #     c = p.createConstraint(robot, parent_joint, robot, joint,
    #                            jointType=p.JOINT_GEAR,
    #                            jointAxis=[0, 0, 1],
    #                            parentFramePosition=unit_point(),
    #                            childFramePosition=unit_point(),
    #                            )
    #     ratio = get_max_limit(robot, parent_joint) / get_max_limit(robot, joint)
    #     #if joint in gripper_joints[2:]:
    #     #    ratio *= -1
    #     if get_joint_name(robot, joint) == 'r_gripper_l_finger_tip_joint':
    #         ratio *= -1
    #     p.changeConstraint(c, gearRatio=-ratio, maxForce=10000)

    # r_gripper_l_finger_tip_joint works with gearRatio=-ratio
    # r_gripper_r_finger_tip_joint doesn't work but gearRatio=-ratio seems best
    # add_gripper_gears(robot, gripper_group)
    for child_joint in robot.get_finger_links(gripper_joints):
        parent_joint = parent_link_from_joint(robot, child_joint)
        ratio = get_max_limit(robot, parent_joint) / get_max_limit(robot, child_joint)
        c = p.createConstraint(
            robot,
            parent_joint,
            robot,
            child_joint,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=unit_point(),
            childFramePosition=unit_point(),
        )
        p.changeConstraint(c, gearRatio=-ratio, maxForce=10000)
        print(
            get_joint_name(robot, parent_joint),
            get_joint_name(robot, child_joint),
            ratio,
        )

        controllable_joints = []
        controllable_joints = [parent_joint]
        # controllable_joints = [child_joint]
        # controllable_joints = [parent_joint, child_joint]
        control_joints(
            robot, controllable_joints, get_max_limits(robot, controllable_joints)
        )

    # control_joint(robot, joint_from_name(robot, 'torso_lift_joint'))
    # control_joint(robot, parent_joint, position=closed_conf[0])

    controller = stall_for_duration(10.0)
    # controller = waypoint_joint_controller(robot, joints, closed_conf)
    simulate_controller(controller)

    wait_if_gui()
    return


def test_mimic():
    add_data_path()
    path = "differential/diff_ring.urdf"
    # path = 'racecar/racecar_differential.urdf'
    body = load_pybullet(path)
    dump_body(body)
    wait_if_gui()
    return


################################################################################

MOUSE_EVENT = {
    # MOUSE_MOVE_EVENT=1, MOUSE_BUTTON_EVENT=2
    "MOUSE_MOVE_EVENT": 1,
    "MOUSE_BUTTON_EVENT": 2,
}
MOUSE_INDEX = {
    # 'button index for left/middle/right mouse button'
    "hover": -1,
    "press": 0,
    # 'left': None,
    # 'middle': None,
    # 'right': None,
}

MOUSE_STATE = {
    # flag KEY_WAS_TRIGGERED /KEY_IS_DOWN /KEY_WAS_RELEASED
    "unchanged": 0,
    "down": 3,
    "released": 4,
    # p.VR_BUTTON_WAS_RELEASED, p.VR_BUTTON_IS_DOWN, p.VR_BUTTON_WAS_TRIGGERED
}

# TODO: p.isConnected, p.getNumUserData


def is_click_event(e):
    return (
        (e.eventType == 2)
        and (e.buttonIndex == 0)
        and (e.buttonState & p.KEY_WAS_TRIGGERED)
    )  # e.buttonState in [2, 3]


def is_press_event(e, key):
    return (key in e) and (
        e[key] & p.KEY_WAS_TRIGGERED
    )  # p.KEY_WAS_RELEASED, p.KEY_IS_DOWN, p.KEY_WAS_TRIGGERED


def random_color_click(mouse_event):
    color = apply_alpha(random_color())
    # step_simulation()
    rayFrom, rayTo = get_ray_from_to(mouse_event.mousePosX, mouse_event.mousePosY)
    # p.addUserDebugLine(rayFrom,rayTo,[1,0,0],3)
    ray_result = RayResult(*p.rayTest(rayFrom, rayTo)[0])
    body = ray_result[0]
    if body == NULL_ID:
        return None
    # p.removeBody(body)
    set_all_color(body, color)
    return body


def test_gui(hz=INF):
    step_simulation()
    set_preview(enable=True)
    # button = add_button('Annotate?')
    threshold = 0.5
    parameter = add_parameter("Annotate (> {:.3f})?".format(threshold))

    key = p.B3G_CONTROL
    # key = ord('q')

    time_step = 1.0 / hz
    num_steps = 0
    time.time()
    keyboard_state = {}
    while p.isConnected():
        time.sleep(time_step)
        num_steps += 1
        keyboard_events = p.getKeyboardEvents()  # dict
        # keyboard_state.update(keyboard_events)
        keyboard_state = keyboard_events
        mouse_events = get_mouse_events()  # list
        if not mouse_events:
            continue
        mouse_event = mouse_events[-1]
        if not is_click_event(mouse_event):
            continue
        print(keyboard_state)
        # print(keyboard_events)
        if (key in keyboard_state) and (keyboard_state[key] == p.KEY_IS_DOWN):
            # if is_press_event(keyboard_events, key):
            continue
        # status = read_button(button)
        status = read_parameter(parameter) > threshold
        print(mouse_event, status)


def create_debug(button_text="goal", parameter_text="scale"):
    button = add_button(button_text)
    parameter = add_parameter(parameter_text)
    return button, parameter


def test_gui_color():
    step_simulation()
    set_preview(enable=True)

    button, parameter = create_debug()
    # p.removeUserDebugItem(button) # Doesn't work on p.addUserDebugParameter
    # p.removeAllUserParameters()
    invert_dict(MOUSE_STATE)

    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/examples/addPlanarReflection.py
    # https://github.com/bulletphysics/bullet3/search?l=Python&q=getRayFromTo&type=
    time_step = 1.0 / 60
    num_steps = 0
    time.time()
    while True:
        # keyboard_events = p.getKeyboardEvents()
        # if keyboard_events:
        #    print('Keyboard | {:.3f} | {:.3f} |'.format(elapsed_time(start_time), num_steps*time_step), keyboard_events)
        mouse_events = get_mouse_events()
        if mouse_events:
            for mouse_event in mouse_events:
                # print(p.getDebugVisualizerCamera()) # Changes only when the window changes
                if mouse_event.buttonState != 0:
                    mouse_event.buttonState
                if is_click_event(mouse_event):
                    random_color_click(mouse_event)
                    # p.removeAllUserParameters() # Removing to reset text
                    # button, parameter = create_debug()
                    print(read_button(button), read_counter(button))
                    print(read_parameter(parameter))

            # if is_click_event(mouse_events[-1]):
            #    print(input('Input:'))

            # print(getRayFromTo(mouse_event.mousePosX, mouse_event.mousePosY))
            # print('Mouse | Click: {} | {:.3f} | {:.3f} |'.format(
            #    name_from_event.get(click, click), elapsed_time(start_time), num_steps*time_step), mouse_events)
        time.sleep(time_step)
        num_steps += 1

    # https://github.com/erwincoumans/bullet3/blob/de8f04f819923a02ca060a2762477365fe216177/examples/pybullet/examples/userData.py
    # MyKey1 = p.addUserData(table, "MyKey1", "MyValue1")
    # print(p.getUserData(MyKey1))


def test_placements(robot, table, region, obj1):
    sampler = get_placement_gen_fn(robot, table)
    region_pose = RelativePose(region)
    generator = sampler(obj1, region, region_pose, shape=None)
    for (pose,) in generator:
        pose.assign()
        wait_for_user()
