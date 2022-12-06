YCB_BANK_DIR = "./vision_utils/ycb_models/"  # path to YCB models used during training of densefusion
POSE_DIR = (
    "./vision_utils/DenseFusion"  # path to densefusion repo (for 6d pose estimation)
)
REAL_WORLD_CLIENT = 0
OBS_TIME_THRESHOLD = 5
DEBUG_DIR = "./debug"

#########################
# maskrcnn params
#########################
MASKRCNN_DIR = "./vision_utils/detectron2"  # path to the folder containing maskrcnn ckpt. don't need to git clone the repo if detectron2 is installed using pip
MASKRCNN_CONFIDENCE_THRESHOLD = 0.7


#########################
# uois params
#########################
UOIS_DIR = "./vision_utils/uois"
dsn_config = {
    # Sizes
    "feature_dim": 64,  # 32 would be normal
    # Mean Shift parameters (for 3D voting)
    "max_GMS_iters": 10,
    "epsilon": 0.05,  # Connected Components parameter
    "sigma": 0.02,  # Gaussian bandwidth parameter
    "num_seeds": 200,  # Used for MeanShift, but not BlurringMeanShift
    "subsample_factor": 5,
    # Misc
    "min_pixels_thresh": 500,
    "tau": 15.0,
}

rrn_config = {
    # Sizes
    "feature_dim": 64,  # 32 would be normal
    "img_H": 224,
    "img_W": 224,
    # architecture parameters
    "use_coordconv": False,
}

uois3d_config = {
    # Padding for RGB Refinement Network
    "padding_percentage": 0.25,
    # Open/Close Morphology for IMP (Initial Mask Processing) module
    "use_open_close_morphology": True,
    "open_close_morphology_ksize": 9,
    # Largest Connected Component for IMP module
    "use_largest_connected_component": True,
}


maskrcnn_class2name = {
    0: "floor",
    1: "table",
    2: "002_master_chef_can",
    3: "003_cracker_box",
    4: "004_sugar_box",
    5: "005_tomato_soup_can",
    6: "006_mustard_bottle",
    7: "007_tuna_fish_can",
    8: "008_pudding_box",
    9: "009_gelatin_box",
    10: "010_potted_meat_can",
    11: "011_banana",
    12: "019_pitcher_base",
    13: "021_bleach_cleanser",
    14: "024_bowl",
    15: "025_mug",
    16: "035_power_drill",
    17: "036_wood_block",
    18: "037_scissors",
    19: "040_large_marker",
    20: "051_large_clamp",
    21: "052_extra_large_clamp",
    22: "061_foam_brick",
}
