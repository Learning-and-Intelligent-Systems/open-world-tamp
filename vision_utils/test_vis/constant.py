import os

VISION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
YCB_BANK_DIR = os.path.join(
    VISION_DIR, "ycb_models/"
)  # path to YCB models used during training of densefusion
POSE_DIR = os.path.join(
    VISION_DIR, "DenseFusion"
)  # path to densefusion repo (for 6d pose estimation)
SEG_DIR = os.path.join(
    VISION_DIR, "detectron2"
)  # path to the folder containing maskrcnn ckpt
REAL_WORLD_CLIENT = 0
OBS_IOU_THRESHOLD = 0.7
OBS_TIME_THRESHOLD = 5
