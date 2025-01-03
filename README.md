# open-world-tamp

In this repo, we implement a general-purpose task and motion planning agent from perceptual input (RGBD Images) and natural language goals. 
This repo builds off the following paper:

Long-Horizon Manipulation of Unknown Objects via Task and Motion Planning with Estimated Affordances
- https://arxiv.org/abs/2108.04145

Please contact **curtisa@mit.edu** before attempting to use it for your own research.

<p align="center">
  <img src="https://github.com/Learning-and-Intelligent-Systems/open-world-tamp/blob/master/figs/rw.gif" alt="animated" width="1024px" />
</p>


## Installation

Clone the repo and its submodules (may take a while due to the large amount of logged data):
```
git clone git@github.com:Learning-and-Intelligent-Systems/open-world-tamp.git
cd open-world-tamp
git submodule update --init --recursive
```

### Dependencies

Install the python dependencies. If possible, install using python3.8 as that appears to be the only python version that supports all of the perceptual dependencies:
```
python -m pip install -e .
```

### FastDownward

Build FastDownward:
```
./tamp/downward/build.py
```

### Segmentation

If you're looking to use the segmentation network with the `--segmentation` flag, you will need to download the pretrained UCN checkpoint from
[here](https://github.com/IRVLUTD/UnseenObjectClustering) and place the checkpoints folder in `vision_utils/ucn/data`


## Estimator, Planner, Controller, and Simulation:
Command line arguments are used to specify the robot, goal, and simulated world.
```
python run_planner.py --robot=pr2 --goal=all_green --world=problem0 -v

python run_planner.py --robot=movo --goal=all_green --world=problem0 -v

python run_planner.py --robot=panda --goal=all_green --world=problem0 -v
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

## Development

### Tests/Coverage
Run the automated tests with the following command
```
pytest tests/
```

Run a coverage test with the following command. You can see the coverage report by opening htmlcov/index.html in a browser.
```
pytest --cov-config=.coveragerc --cov=. --cov-report html tests/
```

