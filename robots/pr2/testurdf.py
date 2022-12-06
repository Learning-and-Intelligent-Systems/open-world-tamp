import sys

sys.path.append("pybullet-planning")

from pybullet_tools.utils import connect, load_pybullet, wait_for_user

connect(use_gui=True)
r = load_pybullet(
    "/home/movo/m0m/open-world-tamp/pybullet-planning/models/movo_description/movo_lis.urdf"
)

wait_for_user()
