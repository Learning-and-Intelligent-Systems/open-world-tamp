# open-world-tamp

## Installation

Clone the repo and its submodules (may take a while due to the large amount of logged data):
```
git clone git@github.mit.edu:Learning-and-Intelligent-Systems/open-world-tamp.git
cd open-world-tamp
git checkout stable
git submodule update --init --recursive
```

### Dependencies

Install the python dependencies. If possible, install using python3.8 as that appears to be the only python version that supports all of the perceptual dependencies:
```
$ python -m pip install -r requirements.txt
```
If you get errors when installing detectron, you may need to modify your paths. Make sure to switch out `11.4` for your current cuda version.
```
export CPATH=/usr/local/cuda-11.4/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.4/bin:$PATH
```

### FastDownward

Build FastDownward:
```
./pddlstream/downward/build.py
```

### IKFast

Compile IKFast:
```
cd pybullet_planning/pybullet_tools/ikfast/<robot-name>
python setup.py
```

## Development

Make sure your submodules are up-to-date:
```
git pull --recurse-submodules
```

### Estimator, Planner, Controller, and Simulation:
Command line arguments are used to specify the robot, goal, and simulated world.
```
python run_planner.py --robot=pr2 --goal=all_green --world=problem0 -v
```

#### Command-line arguments
`--simulated` Specifies if the run is simulated or in the real-world. True by default.

`--world` If simulated, this argument tells OWT how to set up the simulated world

`--segmentation` Specified if the run uses segmentation networks instead of ground truth segmentation from the simulator. False by default.

`--robot` Specifies the robot you are using. Current options are `pr2`, `panda`, `movo`

`--goal` Specifies the objective of the robot. For example, `all_green` instructs the robot to place all movable objects on a green region. 

`-v` Is a flag to visualize the resulting plan. False by default.

`--exploration` Is a flag that specifies if mobile-base exploration should proceed manipulation planning

`--base_planner` Is a flag that specifies the algorithm to use for mobile-base exploration

This is only a subset of the available segmentation flags. See code for more.



## Modules

* Shape completion: https://github.mit.edu/Learning-and-Intelligent-Systems/open-world-tamp/tree/shape_completion/vision_utils
* Grasping: https://github.mit.edu/Learning-and-Intelligent-Systems/open-world-tamp/tree/grasp_new/grasp
