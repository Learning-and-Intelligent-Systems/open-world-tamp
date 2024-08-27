import os

from pybullet_tools.ikfast.pr2.ik import (BASE_FRAME, IK_FRAME, TORSO_JOINT,
                                          UPPER_JOINT)
from pybullet_tools.ikfast.utils import IKFastInfo
from pybullet_tools.utils import get_camera_matrix, list_paths

try:
    import rosgraph

    USING_ROS = rosgraph.is_master_online()
except ImportError:
    USING_ROS = False
print("ROS:", USING_ROS)

try:
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(device)  # cpu | cuda | cuda:0
except ImportError:
    DEVICE = None
print("Device:", DEVICE)

#######################################################

ROOT_DIRECTORY = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, os.pardir)
)
LIS_PR2 = True

PR2_INFOS = {
    arm: IKFastInfo(
        module_name="pr2.ik{}".format(arm.capitalize()),
        base_link=BASE_FRAME,
        ee_link=IK_FRAME[arm],
        free_joints=[TORSO_JOINT, UPPER_JOINT[arm]],
    )
    for arm in IK_FRAME
}

CAMERA_FRAME = "head_mount_kinect_rgb_link"
CAMERA_OPTICAL_FRAME = "head_mount_kinect_rgb_optical_frame"
WIDTH, HEIGHT = 640, 480
FX, FY = 525.0, 525.0
# CX, CY = 319.5, 239.5
CAMERA_MATRIX = get_camera_matrix(WIDTH, HEIGHT, FX, FY)

# CAMERA_FRAME = 'high_def_frame'
# CAMERA_OPTICAL_FRAME = 'high_def_optical_frame' # HEAD_LINK_NAME
# CAMERA_MATRIX = PR2_CAMERA_MATRIX # TODO: WIDTH, HEIGHT

PR2_WINGSPAN = 0.75 * 1.19  # 1.19
PR2_FINGER_DIMENSIONS = [0.02, 0.035]  # width x length
PR2_HAND_DIMENSIONS = [0.085, 0.07]  # width x length

#######################################################

ROOT_PATH = os.path.abspath(os.path.join(__file__, *[os.pardir] * 3))
SRL_PATH = os.path.join(ROOT_PATH, "models/srl")
YCB_PATH = os.path.join(SRL_PATH, "ycb")

LTAMP_PATH = os.path.join(ROOT_PATH, "models/ltamp")
BOWLS_PATH = os.path.join(LTAMP_PATH, "bowls")
CUPS_PATH = os.path.join(LTAMP_PATH, "cups")

GRASPNET_DIR = "data/grasp_dataset"

Z_EPSILON = -1e-3

##########################
# shape completion params
##########################

SC_PATH = "vision_utils/AtlasNet/"  # repo contains both AtlasNet and MSN-PCN utils
CKPT_PATH = "vision_utils/trained_models/msn1.pth"  # AtlasNet(mix | mix2 | mix3) | MSN-PCN(msn1)

#########################
# Mask-RCNN segmentation params
########################
# ----------1-------------
MASKRCNN_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
SEG_CKPT_PATH = "vision_utils/trained_models/maskrcnn_res50_ycb_real+syn.pth"
# SEG_CKPT_PATH = 'vision_utils/trained_models/maskrcnn_rgbd.pth'
# SEG_CKPT_PATH = 'vision_utils/trained_models/maskrcnn_res50_ycb_2.pth'
# -----------2-------------
# MASKRCNN_CONFIG = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
# SEG_CKPT_PATH = 'vision_utils/trained_models/maskrcnn_res101_ycb.pth'
# ------------------------
MASKRCNN_CONFIDENCE_THRESHOLD = 0.7

#########################
# Faster-RCNN detection params
########################
FASTERRCNN_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
DET_CKPT_PATH = "vision_utils/trained_models/bowlcup_detector.pth"
FASTERRCNN_CONFIDENCE_THRESHOLD = 0.7

#########################
# UCN segmentation params
########################

UCN_PATH = "./vision_utils/ucn"
UCN_CKPT_PATH1 = os.path.join(
    UCN_PATH,
    "data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth",
)
UCN_CKPT_PATH2 = os.path.join(
    UCN_PATH,
    "data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth",
)
UCN_CONFIG = os.path.join(
    UCN_PATH, "experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml"
)

#########################
# UOIS segmentation params
########################

UOIS_PATH = "./vision_utils/uois"
UOIS_CKPT_PATH = "./vision_utils/trained_models"


OVERSEG = False
DSN_CONFIG = {
    "feature_dim": 64,
}
DSN_CONFIG.update(
    {
        # overseg
        "min_pixels_thresh": 200,
        "max_GMS_iters": 15,
        "epsilon": 0.03,  # Connected Components parameter
        "sigma": 0.01,  # Gaussian bandwidth parameter for MeanShift(to cluster predicted object center)
        "num_seeds": 500,  # Used for MeanShift, but not BlurringMeanShift
        "subsample_factor": 2,
    }
    if OVERSEG
    else {
        # normal
        "min_pixels_thresh": 500,
        "max_GMS_iters": 10,
        "epsilon": 0.05,  # Connected Components parameter
        "sigma": 0.02,  # Gaussian bandwidth parameter for MeanShift(to cluster predicted object center)
        "num_seeds": 200,  # Used for MeanShift, but not BlurringMeanShift
        "subsample_factor": 5,
    }
)

RRN_CONFIG = {
    "feature_dim": 64,
    "use_coordconv": False,
}

UOIS3D_CONFIG = {
    # Padding for RGB Refinement Network
    "padding_percentage": 0.25,
    # Open/Close Morphology for IMP (Initial Mask Processing) module
    "use_open_close_morphology": True,
    "open_close_morphology_ksize": 6,
    # Largest Connected Component for IMP module
    "use_largest_connected_component": True,
}

##################################################


def ycb_type_from_name(name):
    return name.split("_", 1)[-1]


def ycb_type_from_file(path):
    # TODO: rename to be from_dir
    return ycb_type_from_name(os.path.basename(path))


def get_ycb_obj_path(ycb_type, use_concave=False):
    path_from_type = {
        ycb_type_from_file(path): path
        for path in list_paths(YCB_PATH)
        if os.path.isdir(path)
    }
    # print(ycb_path)
    # for name, path in sorted(path_from_type.items()):
    #     print(name, path)
    if ycb_type not in path_from_type:
        return None
    # texture_map.png textured.mtl
    # filename = 'nontextured.ply' # invalid mesh filename extension '.ply'
    # filename = 'nontextured.stl'
    if use_concave:
        filename = "decomp.obj"
    else:
        filename = "textured.obj"
    # textured_simple.obj.mtl # TODO: where did these originate?
    # filename = 'textured_simple.obj'
    return os.path.join(path_from_type[ycb_type], filename)


#######################################################

# http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
"""
003_cracker_box, 004_sugar_box, 005_tomato_soup_can, 006_mustard_bottle, 008_pudding_box, 009_gelatin_box, 
010_potted_meat_can, 024_bowl, 061_foam_brick

001_chips_can, 002_master_chef_can, 025_mug, 029_plate, 065-a_cups, ..., 065-j_cups
"""

# http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf
YCB_MASSES = {
    # mass of the base, in kg (if using SI units)
    "cracker_box": 0.411,
    "apple": 0.068,
    "sugar_box": 0.514,
    "tomato_soup_can": 0.349,
    "mustard_bottle": 0.603,
    "pudding_box": 0.187,
    "gelatin_box": 0.097,
    "potted_meat_can": 0.370,
    "bowl": 0.147,
    "foam_brick": 0.028,
    "lemon": 0.029,
    "sponge": 0.006,
    "orange": 0.047,
    "spatula": 0.051,
    "skillet": 0.951,
    "fork": 0.034,
    "hammer": 0.665,
    "strawberry": 0.018,
    "padlock": 0.208,
    "knife": 0.031,
    "bleach_cleanser": 1.131,
    "pear": 0.049,
    "pitcher_base": 0.178,
    "scissors": 0.082,
    "master_chef_can": 0.414,
    "spoon": 0.030,
    "chips_can": 0.205,
    "peach": 0.033,
    "banana": 0.066,
    "plum": 0.025,
    "tuna_fish_can": 0.171,
    "plate": 0.279,
    "wood_block": 0.729,
    "mug": 0.118,
    "power_drill": 0.895,
}

YCB_COLORS = {
    "bowl": (0.4, 0.4, 0.6, 1),
    "mug": (0.7, 0.2, 0.2, 1),
}

YCB_DIMENSIONS = {
    # Dimensions (mm)
    "cracker_box": [60, 158, 210],
    "potted_meat_can": [50, 97, 82],
    # TODO: add the other objects
}

LIS_YCB = [
    # Relevant YCB objects currently in the lab
    "002_master_chef_can",  # i.e. coffee
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    #'010_potted_meat_can', # Missing
    "021_bleach_cleanser",  # i.e. scrub cleanser
    #'024_bowl', # Missing
    "025_mug",
    #'027-skillet', # No segmentation
    #'029_plate', # No segmentation
    "036_wood_block",
    "061_foam_brick",
    #'065-a_cups ', # a to j, no segmentation
    #'070-a_colored_wood_blocks' # No segmentation
    #'077_rubiks_cube' # Missing, no segmentation
]

#######################################################

# def sample_parameters(randomize=True):
#     # TODO: fix these values per object
#     scale = int(randomize)
#     return {
#         # TODO: randomize masses here
#         # lateral (linear) contact friction
#         # The coefficient of friction is usually between 0 and 1 but can be greater than 1
#         # A coefficient of friction that is more than one just means that the frictional force is stronger than the normal force
#         'lateralFriction': sample_norm(mu=0.5, sigma=0.1 * scale, lower=0.0),
#         # torsional friction around the contact normal
#         'spinningFriction': sample_norm(mu=0.001, sigma=0.0005 * scale, lower=0.0),
#         # torsional friction orthogonal to contact normal
#         # keep this value very close to zero, otherwise the simulation can become very unrealistic
#         'rollingFriction': sample_norm(mu=0.001, sigma=0.0005 * scale, lower=0.0),
#         # bouncyness of contact. Keep it a bit less than 1.
#         # restitution: 0 => inelastic collision, 1 => elastic collision
#         'restitution': sample_norm(mu=0.7, sigma=0.1 * scale, lower=0.0, upper=1.0),
#         # linear damping of the link (0.04 by default)
#         'linearDamping': sample_norm(mu=0.04, sigma=0.01 * scale, lower=0.01),
#         # angular damping of the link (0.04 by default)
#         'angularDamping': sample_norm(mu=0.04, sigma=0.01 * scale, lower=0.01),
#         #'contactStiffness': None,
#         #'contactDamping': None,
#         #'localInertiaDiagonal': (sample_norm(mu=5e-9, sigma=1e-9, lower=1e-9) * np.ones(3)).tolist(),
#     }
