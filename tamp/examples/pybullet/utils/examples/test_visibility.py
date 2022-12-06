#!/usr/bin/env python

from __future__ import print_function

import pybullet as p

from pybullet_tools.pr2_utils import HEAD_LINK_NAME, PR2_GROUPS, get_viewcone, get_detections, \
    REST_LEFT_ARM, rightarm_from_leftarm, inverse_visibility, get_detection_cone, visible_base_generator, \
    DRAKE_PR2_URDF
from pybullet_tools.utils import joint_from_name, set_joint_position, disconnect, HideOutput, \
    set_joint_positions, connect, wait_if_gui, get_link_pose, link_from_name, set_point, set_pose, \
    dump_body, load_model, create_mesh, point_from_pose, get_pose, joints_from_names, BLOCK_URDF, \
    remove_body, child_link_from_joint, RED, BLUE, get_link_name, add_line, draw_point


def main():
    # TODO: update this example

    connect(use_gui=True)
    with HideOutput():
        pr2 = load_model(DRAKE_PR2_URDF)
    set_joint_positions(pr2, joints_from_names(pr2, PR2_GROUPS['left_arm']), REST_LEFT_ARM)
    set_joint_positions(pr2, joints_from_names(pr2, PR2_GROUPS['right_arm']), rightarm_from_leftarm(REST_LEFT_ARM))
    set_joint_positions(pr2, joints_from_names(pr2, PR2_GROUPS['torso']), [0.2])
    dump_body(pr2)

    block = load_model(BLOCK_URDF, fixed_base=False)
    set_point(block, [2, 0.5, 1])
    target_point = point_from_pose(get_pose(block))
    draw_point(target_point)

    head_joints = joints_from_names(pr2, PR2_GROUPS['head'])
    #head_link = child_link_from_joint(head_joints[-1])
    #head_name = get_link_name(pr2, head_link)

    head_name = 'high_def_optical_frame' # HEAD_LINK_NAME | high_def_optical_frame | high_def_frame
    head_link = link_from_name(pr2, head_name)

    #max_detect_distance = 2.5
    max_register_distance = 1.0
    distance_range = (max_register_distance/2, max_register_distance)
    base_generator = visible_base_generator(pr2, target_point, distance_range)

    base_joints = joints_from_names(pr2, PR2_GROUPS['base'])
    for i in range(5):
        base_conf = next(base_generator)
        set_joint_positions(pr2, base_joints, base_conf)

        handles = [
            add_line(point_from_pose(get_link_pose(pr2, head_link)), target_point, color=RED),
            add_line(point_from_pose(get_link_pose(pr2, link_from_name(pr2, HEAD_LINK_NAME))), target_point, color=BLUE),
        ]

        # head_conf = sub_inverse_kinematics(pr2, head_joints[0], HEAD_LINK, )
        head_conf = inverse_visibility(pr2, target_point, head_name=head_name, head_joints=head_joints)
        assert head_conf is not None
        set_joint_positions(pr2, head_joints, head_conf)
        print(get_detections(pr2))
        # TODO: does this detect the robot sometimes?

        detect_mesh, z = get_detection_cone(pr2, block)
        detect_cone = create_mesh(detect_mesh, color=(0, 1, 0, 0.5))
        set_pose(detect_cone, get_link_pose(pr2, link_from_name(pr2, HEAD_LINK_NAME)))
        view_cone = get_viewcone(depth=2.5, color=(1, 0, 0, 0.25))
        set_pose(view_cone, get_link_pose(pr2, link_from_name(pr2, HEAD_LINK_NAME)))
        wait_if_gui()
        remove_body(detect_cone)
        remove_body(view_cone)

    disconnect()

if __name__ == '__main__':
    main()