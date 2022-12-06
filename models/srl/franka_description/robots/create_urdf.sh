#!/bin/bash

source /opt/ros/kinetic/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# add to ROS package path
export ROS_PACKAGE_PATH=${SCRIPT_DIR}/../..:$ROS_PACKAGE_PATH

echo "Using ROS_PACKAGE_PATH ${ROS_PACKAGE_PATH}"

rosrun xacro xacro --inorder -o panda_arm.urdf panda_arm.urdf.xacro
rosrun xacro xacro --inorder -o panda_arm_hand.urdf panda_arm_hand.urdf.xacro
rosrun xacro xacro --inorder -o panda_arm_hand_on_carter.urdf panda_arm_hand_on_carter.urdf.xacro

# body options
export SEGWAY_HAS_FRONT_CASTER=true
export SEGWAY_HAS_REAR_CASTER=true

# accessories
#export SEGWAY_HAS_EXT_IMU=false
#export SEGWAY_HAS_GPS=false
#export SEGWAY_HAS_KINECT=false
#export SEGWAY_HAS_ONE_2D_LASER=false
#export SEGWAY_LASER1_IS_HOKUYO=false
#export SEGWAY_HAS_SECOND_2D_LASER=false
#export SEGWAY_LASER2_IS_HOKUYO=false
#export SEGWAY_HAS_VLP16=false
#export SEGWAY_HAS_FLEA3=false

rosrun xacro xacro --inorder -o panda_arm_hand_on_segway.urdf panda_arm_hand_on_segway.urdf.xacro

