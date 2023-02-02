# open-world-tamp
<p align="center">
  <img src="https://github.com/Learning-and-Intelligent-Systems/open-world-tamp/blob/master/figs/rw.gif" alt="animated" width="1024px" />
</p>


## Installation

Clone the repo and its submodules (may take a while due to the large amount of logged data):
```
git clone git@github.com:Learning-and-Intelligent-Systems/open-world-tamp.git
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
./tamp/downward/build.py
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

## Segmentation

If you're looking to use the segmentation network with the `--segmentation` flag, you will need to download the pretrained UCN checkpoint from
[here](https://drive.google.com/file/d/1O-ymMGD_qDEtYxRU19zSv17Lgg6fSinQ/view) and place the checkpoints folder in `vision_utils/ucn/data`


## Estimator, Planner, Controller, and Simulation:
Command line arguments are used to specify the robot, goal, and simulated world.
```
python run_planner.py --robot=pr2 --goal=all_green --world=problem0 -v
```


<p align="center">
  <img src="https://github.com/Learning-and-Intelligent-Systems/open-world-tamp/blob/master/figs/sims.gif" alt="animated"  width="1024px"  />
</p>



### Command-line arguments
`--simulated` Specifies if the run is simulated or in the real-world. True by default.

`--world` If simulated, this argument tells OWT how to set up the simulated world

`--segmentation` Specified if the run uses segmentation networks instead of ground truth segmentation from the simulator. False by default.

`--robot` Specifies the robot you are using. Current options are `pr2`, `panda`, `movo`

`--goal` Specifies the objective of the robot. For example, `all_green` instructs the robot to place all movable objects on a green region. 

`-v` Is a flag to visualize the resulting plan. False by default.

`--real` Is a flag that specifies if mobile-base exploration should proceed manipulation planning

`--base_planner` Is a flag that specifies the algorithm to use for mobile-base exploration

This is only a subset of the available segmentation flags. See code for more.

### Exploration
Combine mobile-base exploration with fixed-based manipulation by calling the planner with the following flags

`--exploration` Toggles exploration

`--base_planner` Selects the planner to use for exploration. Default is VA*, but more advanced planners can also be used.


<p align="center">
  <img src="https://github.com/Learning-and-Intelligent-Systems/open-world-tamp/blob/master/figs/mb.gif" alt="animated"  width="1024px"  />
</p>

