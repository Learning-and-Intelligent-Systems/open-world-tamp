import os
import sys
from collections import Counter, OrderedDict, namedtuple
from itertools import product
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pybullet_tools.utils import implies

from open_world.estimation.belief import take_ml_estimate
from open_world.simulation.entities import BG, BOWL, CUP, OTHER, TABLE, UNKNOWN
from open_world.simulation.lis import (
    CKPT_PATH,
    DET_CKPT_PATH,
    DEVICE,
    DSN_CONFIG,
    FASTERRCNN_CONFIDENCE_THRESHOLD,
    FASTERRCNN_CONFIG,
    MASKRCNN_CONFIDENCE_THRESHOLD,
    MASKRCNN_CONFIG,
    RRN_CONFIG,
    SC_PATH,
    SEG_CKPT_PATH,
    UCN_CKPT_PATH1,
    UCN_CKPT_PATH2,
    UCN_CONFIG,
    UCN_PATH,
    UOIS3D_CONFIG,
    UOIS_CKPT_PATH,
    UOIS_PATH,
    YCB_MASSES,
    ycb_type_from_name,
)

DEFAULT_DEBUG = False
FLOOR = 0
TABLE_IDNUM = 1
BACKGROUND = [FLOOR, TABLE_IDNUM]  # 0 - floor, 1 - table
# TODO clean up constants
INSTANCE_TEMPLATE = "instance_{}"
DEFAULT_VALUE = (UNKNOWN, UNKNOWN)

YCB_CLASSES = [
    BG,
    TABLE,
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
]

FASTERRCNN_CLASS = [
    # original classes: 0-bowl, 1-cup. modified for compatibility w/ other networks
    BG,
    TABLE,
    BOWL,
    CUP,
]

#######################################################

AtlasOpt = namedtuple(
    "AtlasOpt",
    [
        "demo",
        "SVR",
        "reload_model_path",
        "nb_primitives",
        "template_type",
        "dim_template",
        "device",
        "bottleneck_size",
        "number_points",
        "number_points_eval",
        "remove_all_batchNorms",
        "hidden_neurons",
        "num_layers",
        "activation",
    ],
)


def init_atlas(ckpt_path):

    # TODO(xiaolin)
    # RuntimeError: Error(s) in loading state_dict for EncoderDecoder:
    #         Missing key(s) in state_dict: "encoder.conv1.weight", "encoder.conv1.bias",

    from model.atlasnet import Atlasnet
    from model.model_blocks import (
        PointNet,
    )  # NOTE: if used together with DenseFusion, be careful about the class name

    class EncoderDecoder(nn.Module):
        def __init__(self, opt):
            super(EncoderDecoder, self).__init__()
            self.encoder = PointNet(nlatent=opt.bottleneck_size)
            self.decoder = Atlasnet(opt)
            self.to(opt.device)
            self.eval()
            self.device = opt.device

        def forward(self, x, train=True):  # refinement
            return self.decoder(self.encoder(x), train=train)

        def generate_mesh(self, x):
            atlas_list = self.decoder.generate_mesh(self.encoder(x))
            return atlas_list  # a list of atlas, each can be converted to a mesh
            # return self.decoder.generate_mesh(self.encoder(x))

    opt = AtlasOpt(
        True,
        True,
        ckpt_path,
        1,
        "SPHERE",
        3,
        DEVICE,
        1024,
        2500,
        2500,
        False,
        512,
        2,
        "relu",
    )
    network = EncoderDecoder(opt)
    sdict = torch.load(opt.reload_model_path, map_location=DEVICE)
    # resolve key name issue in multiGPU
    new_dict = OrderedDict()
    for k, v in sdict.items():
        name = k[7:]
        new_dict[name] = v
    network.load_state_dict(new_dict)
    return network


def init_msn(ckpt_path):
    from model.model_blocks import PointGenCon, PointNetfeat, PointNetRes

    class MSN(nn.Module):
        # modified for testing purpose (remove expansion & MDS)
        def __init__(
            self,
            num_points=8192,
            bottleneck_size=1024,
            n_primitives=16,
            device=None,
            use_template=False,
        ):
            super(MSN, self).__init__()
            self.num_points = num_points
            self.bottleneck_size = bottleneck_size
            self.n_primitives = n_primitives
            self.encoder = nn.Sequential(
                PointNetfeat(num_points, global_feat=True),
                nn.Linear(1024, self.bottleneck_size),
                nn.BatchNorm1d(self.bottleneck_size),
                nn.ReLU(),
            )
            self.decoder = nn.ModuleList(
                [
                    PointGenCon(bottleneck_size=2 + self.bottleneck_size)
                    for i in range(0, self.n_primitives)
                ]
            )
            self.res = PointNetRes()
            self.device = device

            self.use_template = use_template
            if use_template:  # use rectangle template during inference
                import meshzoo

                # from model.template import get_template # borrow the template functions from AtlasNet
                self.vertices, self.faces = meshzoo.rectangle(
                    nx=23, ny=23
                )  # (529,2),(968,3)
                self.vertices = self.vertices.transpose(1, 0)[
                    np.newaxis, ...
                ]  # (1, 2, 529)

        def forward(self, x):
            partial = x
            x = self.encoder(x)
            outs = []
            for i in range(0, self.n_primitives):
                if self.use_template:
                    y = (
                        x.unsqueeze(2)
                        .expand(x.size(0), x.size(1), self.vertices.shape[2])
                        .contiguous()
                    )
                    y = torch.cat(
                        (
                            torch.FloatTensor(self.vertices)
                            .repeat(x.size(0), 1, 1)
                            .to(x.device),
                            y,
                        ),
                        1,
                    ).contiguous()
                    outs.append(self.decoder[i](y))  # (1, 3, 529)
                else:
                    rand_grid = torch.FloatTensor(
                        x.size(0), 2, self.num_points // self.n_primitives
                    ).to(x.device)
                    rand_grid.data.uniform_(0, 1)
                    y = (
                        x.unsqueeze(2)
                        .expand(x.size(0), x.size(1), rand_grid.size(2))
                        .contiguous()
                    )
                    y = torch.cat((rand_grid, y), 1).contiguous()
                    outs.append(self.decoder[i](y))
            # return outs
            outs = torch.cat(outs, 2).contiguous()  # 1(batch_size) x 3 x KN
            outs.transpose(1, 2).contiguous()

            id0 = torch.zeros(outs.shape[0], 1, outs.shape[2]).to(x.device).contiguous()
            outs = torch.cat((outs, id0), 1)
            id1 = (
                torch.ones(partial.shape[0], 1, partial.shape[2])
                .to(x.device)
                .contiguous()
            )
            partial = torch.cat((partial, id1), 1)
            xx = torch.cat((outs, partial), 2)

            delta = self.res(xx)
            xx = xx[:, 0:3, :]
            out2 = (xx + delta).transpose(2, 1).contiguous()
            # step = out2.shape[1]//16
            # return [out2.transpose(1,2)[:,:,i*step:(i+1)*step] for i in range(16)]
            return out2.transpose(1, 2).unsqueeze(0)

    network = MSN(num_points=2500, n_primitives=16, device=DEVICE)
    network.to(DEVICE)
    sdict = torch.load(ckpt_path, map_location=DEVICE)
    # resolve key name issue in multiGPU
    # TODO double check key name in checkpoints
    new_dict = OrderedDict()
    for k, v in sdict.items():
        if "stn" in k:  # deprecated module. weight not used
            continue
        name = k  # [7:]
        new_dict[name] = v
    network.load_state_dict(new_dict)
    network.eval()
    return network


def init_sc(base_path=None, ckpt_path=None, branch="msn"):
    """
    initialize shape completion network . atlas | msn

    base_path: root dir to vision repositories
    ckpt_path: path to pretrained model (.pth)

    """
    base_path = base_path if base_path is not None else SC_PATH
    sys.path.append(base_path)
    ckpt_path = ckpt_path if ckpt_path is not None else CKPT_PATH

    if branch == "atlas":
        return init_atlas(ckpt_path)
    elif branch == "msn":
        return init_msn(ckpt_path)
    raise NotImplementedError(branch)


#######################################################


def get_class_frequencies(classes, masks):
    # TODO: apply elsewhere
    return Counter(
        {YCB_CLASSES[cls]: np.count_nonzero(mask) for cls, mask in zip(classes, masks)}
    )


def iterate_array(array, dims=None):
    if dims is None:
        dims = range(len(array.shape))
    dims = list(dims)
    assert set(dims) <= set(range(len(array.shape)))  # TODO: check for repeats
    return product(*(range(array.shape[d]) for d in dims))
    # return np.ndindex(*(array.shape[d] for d in dims))


def str_from_int_seg(int_seg, only_known=False):
    str_seg = np.full(
        int_seg.shape[:2] + (len(DEFAULT_VALUE),), DEFAULT_VALUE, dtype=object
    )  # H x W x 2
    for r, c in iterate_array(str_seg, dims=[0, 1]):
        cls, i = int_seg[r, c, :]
        # str_seg[masks[i], ...] = [cls, BASE_LINK]
        # if cls in BACKGROUND: # skip floor and table
        if cls == FLOOR:  # skip background
            continue
        if cls >= len(YCB_CLASSES):
            category = OTHER
        else:
            category = ycb_type_from_name(YCB_CLASSES[cls])
        if implies(only_known, category in YCB_MASSES):
            # TODO: alternatively check YCB_PATH
            str_seg[r, c, 0] = category
        str_seg[r, c, 1] = INSTANCE_TEMPLATE.format(i)
    return str_seg


def str_from_int_seg_fasterrcnn(int_seg, **kwargs):
    str_seg = np.full(
        int_seg.shape[:2] + (len(DEFAULT_VALUE),), DEFAULT_VALUE, dtype=object
    )  # H x W x 2
    for r, c in iterate_array(str_seg, dims=[0, 1]):
        cls, i = int_seg[r, c, :]
        if cls == FLOOR:  # skip background
            continue
        if cls >= len(FASTERRCNN_CLASS):
            category = OTHER
        else:
            category = FASTERRCNN_CLASS[cls]
        str_seg[r, c, 0] = category
        str_seg[r, c, 1] = INSTANCE_TEMPLATE.format(i)
    return str_seg


def str_from_int_seg_general(int_seg, use_classifer=False, **kwargs):
    if use_classifer:
        return str_from_int_seg_fasterrcnn(int_seg, **kwargs)
    return str_from_int_seg(int_seg, **kwargs)


#######################################################


class MaskRCNN(object):
    def __init__(self, maskrcnn_rgbd=False, **kwargs):
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        """ Detection & Segmentation - MaskRCNN """
        cfg = get_cfg()
        cfg.MODEL.DEVICE = DEVICE.type  # string. cpu|cuda
        cfg.merge_from_file(model_zoo.get_config_file(MASKRCNN_CONFIG))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 23  # len(YCB_CLASSES)
        cfg.MODEL.WEIGHTS = SEG_CKPT_PATH
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MASKRCNN_CONFIDENCE_THRESHOLD  # Only return detections with a confidence score exceeding this threshold
        if maskrcnn_rgbd:
            # imagnet mean&std in BGR order
            cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 0.0]
            cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]  # TODO latest ckpt fix
            # cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395, 1.0]
        self.use_rgbd = maskrcnn_rgbd
        self.predictor = DefaultPredictor(cfg)

        # forward hook + partial execution to obtain confidence for all classes
        self.intermediate_result = {}

        def proposal_hook(module, inp, output):
            # detectron2/modeling/roi_heads/roi_heads.py#L673
            assert len(inp) == 4
            self.intermediate_result["proposals"] = inp[2]

        def boxpredctions_hook(module, inp, output):
            # a) detectron2/modeling/roi_heads/roi_heads.py#L749
            # b) detectron2/modeling/roi_heads/fast_rcnn.py#L432
            # 'predictions' of (a) is output of (b)
            self.intermediate_result[
                "box_predictions"
            ] = output  # output = (scores, proposal_deltas)

        self.predictor.model.roi_heads.register_forward_hook(proposal_hook)
        self.predictor.model.roi_heads.box_predictor.register_forward_hook(
            boxpredctions_hook
        )

    def get_intermediate_result(self):
        _, select_idx = self.predictor.model.roi_heads.box_predictor.inference(
            self.intermediate_result["box_predictions"],
            self.intermediate_result["proposals"],
        )
        boxes = self.predictor.model.roi_heads.box_predictor.predict_boxes(
            self.intermediate_result["box_predictions"],
            self.intermediate_result["proposals"],
        )  # 1000 x 23 x 4
        scores = self.predictor.model.roi_heads.box_predictor.predict_probs(
            self.intermediate_result["box_predictions"],
            self.intermediate_result["proposals"],
        )  # 1000 x 24
        valid_mask = torch.isfinite(boxes[0]).all(dim=1) & torch.isfinite(
            scores[0]
        ).all(dim=1)
        if not valid_mask.all():
            scores = scores[valid_mask]
        cls_scores = (
            scores[0].detach().cpu().numpy()[select_idx[0].detach().cpu().numpy()]
        )
        self.intermediate_result = {}  # reset
        return cls_scores

    def relabel_category(self, assumed_categories, pred_classes):
        name2id = {
            ycb_type_from_name(YCB_CLASSES[i]): i for i in range(len(YCB_CLASSES))
        }
        assumed_category_ids = [name2id[a_cls] for a_cls in assumed_categories]
        if not all([a_cls_id in pred_classes for a_cls_id in assumed_category_ids]):
            # TODO assume #detection >= #assumed categories
            confidence_allclasses = self.get_intermediate_result()  # K x (23 + 1)
            cls_obj = (~np.isin(pred_classes, [0, 1, 23])).nonzero()[0]  # bg, table
            pred_classes[cls_obj] = take_ml_estimate(
                confidence_allclasses[cls_obj], assumed_category_ids
            )
        return pred_classes

    def erode(self, masks, kernel_size=3):
        from scipy.ndimage import binary_erosion

        for i in range(masks.shape[0]):
            masks[i] = binary_erosion(
                masks[i], structure=np.ones((kernel_size, kernel_size))
            ).astype(masks.dtype)
        return masks

    def get_seg(
        self,
        rgb_image,
        only_known=False,
        debug=DEFAULT_DEBUG,
        filter_overlap=True,
        verbose=True,
        return_int=False,
        assumption={},
        conservative=False,
        depth_image=None,
        **kwargs
    ):
        if self.use_rgbd:
            assert depth_image is not None
            output_raw = self.predictor(
                np.concatenate(
                    (rgb_image[..., ::-1], depth_image[..., np.newaxis]), axis=2
                )
            )  # BGRD
        else:
            output_raw = self.predictor(
                rgb_image[:, :, ::-1].astype(np.uint8)
            )  # NOTE: maskrcnn is trained with BGR images. img in param is RGB.
        # Instances - num_instances, image_height, image_width, fields=[pred_boxes, scores, pred_classes, pred_masks]
        masks = output_raw["instances"].pred_masks.detach().cpu().numpy()
        classes = (
            output_raw["instances"].pred_classes.detach().cpu().numpy()
        )  # cls of each detected obj. K
        confidence = output_raw["instances"].scores.detach().cpu().numpy()

        if conservative:
            masks = self.erode(masks)

        if verbose:
            print("Segmentation:", get_class_frequencies(classes, masks))

        if "category" in assumption.keys():
            classes = self.relabel_category(assumption["category"], classes)

        if (len(masks) >= 1) and filter_overlap:
            masks, valid_masks = self.filter_masks(masks, confidence)
            classes = classes[valid_masks]
            confidence = confidence[valid_masks]
            if verbose:
                print("Filtered segmentation:", get_class_frequencies(classes, masks))

        segment = np.zeros(rgb_image.shape[:2] + (2,), dtype=int)  # H x W x 2
        for i, cls in sorted(
            enumerate(classes), key=itemgetter(1)
        ):  # in order of confidence
            # important that the table comes before the rest
            if cls == FLOOR:
                continue  # skip background

            # TODO: ensure table has i == 1
            segment[masks[i], 0] = cls
            segment[masks[i], 1] = (
                (i + 2) if cls != 1 else 1
            )  # 0 - bg, 1 - table, 2+ - object
        if not return_int:
            segment = str_from_int_seg(segment, only_known=only_known)
        if debug:  # visualize the input rgb image and mask
            self.vis(rgb_image, output_raw, **kwargs)
        return segment

    def filter_masks(self, masks, confidences, iou_threshold=0.7):
        """
        filter overlapped masks.
        if iou(mask_A, mask_B)>threshold, return the one with higher confidence.

        masks: K x H x W
        confidences: K
        """
        num_detect = masks.shape[0]
        assert confidences.shape[0] == num_detect
        filtered_masks = []
        invalid_idx = []
        for i in range(num_detect):  # TODO use something better than greedy
            if i in invalid_idx:
                continue
            highest_iou = 0.0
            h_idx = -1
            for j in range(i + 1, num_detect):
                if j in invalid_idx:
                    continue
                iou = (
                    np.logical_and(masks[i], masks[j]).sum()
                    / np.logical_or(masks[i], masks[j]).sum()
                )
                if iou > highest_iou:
                    highest_iou = iou
                    h_idx = j
            if highest_iou > iou_threshold:
                high_conf = i if confidences[i] > confidences[h_idx] else h_idx
                filtered_masks.append(high_conf)
                invalid_idx.append(h_idx)
            else:
                filtered_masks.append(i)
        valid_masks = np.concatenate(
            [masks[i][np.newaxis, ...] for i in filtered_masks]
        )
        return valid_masks, filtered_masks

    def vis(self, rgb_image, outputs, return_ax=False, **kwargs):
        from detectron2.data import MetadataCatalog
        from detectron2.utils.visualizer import ColorMode, Visualizer

        from vision_utils.test_vis_clean.constant import maskrcnn_class2name

        MetadataCatalog.get("tmp").set(thing_classes=list(maskrcnn_class2name.values()))
        sync_metadata = MetadataCatalog.get("tmp")
        v = Visualizer(
            rgb_image,
            metadata=sync_metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW,
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        import matplotlib.pyplot as plt

        if return_ax:
            plt.figure(figsize=(14, 14))
            plt.subplot(2, 2, 1)
            plt.axis("off")
            plt.imshow(rgb_image)
            plt.subplot(2, 2, 2)
            plt.axis("off")
            plt.imshow(v.get_image())
            plt.title(self.__class__.__name__.lower())
        else:
            plt.figure(figsize=(14, 7))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.imshow(rgb_image)
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.imshow(v.get_image())
            # plt.show()
            # plt.close()
        MetadataCatalog.remove("tmp")


class FasterRCNN(object):
    def __init__(self, **kwargs):
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.MODEL.DEVICE = DEVICE.type  # string. cpu|cuda
        cfg.merge_from_file(model_zoo.get_config_file(FASTERRCNN_CONFIG))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 0 - bowl, 1 - cup
        cfg.MODEL.WEIGHTS = DET_CKPT_PATH
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = FASTERRCNN_CONFIDENCE_THRESHOLD  # Only return detections with a confidence score exceeding this threshold
        self.predictor = DefaultPredictor(cfg)

    def relabel_category(self, assumed_categories, pred_classes):
        name2id = {
            ycb_type_from_name(YCB_CLASSES[i]): i for i in range(len(YCB_CLASSES))
        }
        assumed_category_ids = [name2id[a_cls] for a_cls in assumed_categories]
        if not all([a_cls_id in pred_classes for a_cls_id in assumed_category_ids]):
            # TODO assume #detection >= #assumed categories
            confidence_allclasses = self.get_intermediate_result()  # K x (23 + 1)
            cls_obj = (~np.isin(pred_classes, [0, 1, 23])).nonzero()[0]  # bg, table
            pred_classes[cls_obj] = take_ml_estimate(
                confidence_allclasses[cls_obj], assumed_category_ids
            )
        return pred_classes

    def bbox2mask(self, bbox, im_h, im_w):
        mask = np.zeros((im_h, im_w))
        x1, y1, x2, y2 = list(map(int, bbox))
        mask[y1:y2, x1:x2] = 1
        return mask > 0  # boolean

    def get_bbox(
        self,
        rgb_image,
        debug=DEFAULT_DEBUG,
        verbose=True,
        return_int=True,
        assumption={},
        **kwargs
    ):
        output_raw = self.predictor(
            rgb_image[:, :, ::-1]
        )  # NOTE: maskrcnn is trained with BGR images. img in param is RGB.
        # Instances - num_instances, image_height, image_width, fields=[pred_boxes, scores, pred_classes, pred_masks]
        bboxes = output_raw["instances"].pred_boxes.tensor.detach().cpu().numpy()
        classes = (
            output_raw["instances"].pred_classes.detach().cpu().numpy()
        )  # cls of each detected obj. K
        # confidence = output_raw['instances'].scores.detach().cpu().numpy()

        # original classes: 0-bowl, 1-cup. modified for compatibility w/ other networks
        classes = classes + len(BACKGROUND)
        if verbose:
            print("Detections:", list(map(lambda x: FASTERRCNN_CLASS[x], classes)))

        # classes = self.relabel_category(assumption['category'], classes, confidence)
        segment = np.zeros(rgb_image.shape[:2] + (2,), dtype=int)  # H x W x 2
        for i, cls in enumerate(classes):  # in order of confidence
            mask = self.bbox2mask(bboxes[i], *rgb_image.shape[:2])
            segment[mask, 0] = cls
            segment[mask, 1] = i + 2  # obj id starts from 2
        if not return_int:
            segment = str_from_int_seg_fasterrcnn(segment)
        if debug:  # visualize the bbox
            self.vis(rgb_image, output_raw, **kwargs)
        return segment

    def vis(self, rgb_image, outputs, **kwargs):
        from detectron2.data import MetadataCatalog
        from detectron2.utils.visualizer import ColorMode, Visualizer

        MetadataCatalog.get("tmp").set(
            thing_classes=FASTERRCNN_CLASS[len(BACKGROUND) :]
        )
        sync_metadata = MetadataCatalog.get("tmp")
        v = Visualizer(
            rgb_image,
            metadata=sync_metadata,
            scale=1.0,
            instance_mode=ColorMode.SEGMENTATION,
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        import matplotlib.pyplot as plt

        if True:  # subplot
            plt.figure(figsize=(14, 14))
            plt.subplot(2, 2, 1)
            plt.axis("off")
            plt.imshow(rgb_image)
            plt.subplot(2, 2, 2)
            plt.axis("off")
            plt.imshow(v.get_image())
            plt.title(self.__class__.__name__.lower())
        MetadataCatalog.remove("tmp")


class CategoryAgnosticSeg(object):
    """
    UCN or UOIS
    """

    # common stat from ImageNet
    image_net_mean = [0.485, 0.456, 0.406]
    image_net_std = [0.229, 0.224, 0.225]

    def __init__(self):
        pass

    def erode(self, masks, kernel_size=3):
        from scipy.ndimage import binary_erosion

        for i in np.unique(masks):
            if i in BACKGROUND:
                continue
            mask = masks == i
            boundary_mask = np.logical_xor(
                binary_erosion(mask, structure=np.ones((kernel_size, kernel_size))),
                mask,
            )
            masks[boundary_mask] = TABLE_IDNUM
        return masks

    def vis_plot(self, rgb_image, seg_mask, return_ax=False, save_fig=False, **kwargs):
        from detectron2.structures import Instances
        from detectron2.utils.visualizer import ColorMode, Visualizer

        pred_objs = np.unique(seg_mask)
        pred_objs = pred_objs[~np.isin(pred_objs, BACKGROUND)]

        from detectron2.structures import Instances
        from detectron2.utils.colormap import random_color
        from detectron2.utils.visualizer import ColorMode, GenericMask, Visualizer

        v = Visualizer(rgb_image, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        result_wrap = Instances(
            rgb_image.shape[:2],
            pred_masks=torch.BoolTensor([seg_mask == obj_id for obj_id in pred_objs]),
        )
        masks = np.asarray(result_wrap.pred_masks)
        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
        alpha = 0.6
        masks = v._convert_masks(masks)
        areas = np.asarray([x.area() for x in masks])
        sorted_idxs = np.argsort(-areas).tolist()
        masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
        num_instances = len(masks)
        assigned_colors = [
            random_color(rgb=True, maximum=1) for _ in range(num_instances)
        ]
        for ki in range(num_instances):
            color = assigned_colors[ki]
            if masks is not None:
                for segment in masks[ki].polygons:
                    v.draw_polygon(
                        segment.reshape(-1, 2), color, alpha=alpha, edge_color="cyan"
                    )
        # change edgecolor or alpha value for better visualization
        v = v.output

        if save_fig:
            self.fig = v.get_image()
            return
        import matplotlib.pyplot as plt

        if return_ax:
            plt.subplot(2, 2, 3)
            plt.axis("off")
            plt.imshow(v.get_image())
            plt.title(self.__class__.__name__.lower())
        else:
            plt.figure(figsize=(14, 7))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.imshow(rgb_image)
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.imshow(v.get_image())
            # plt.show()
            # plt.close()

    def vis(
        self,
        seg_mask,
        fg_mask,
        center_offset,
        initial_mask,
        batch,
        return_ax=False,
        **kwargs
    ):
        from src.util.utilities import get_color_mask  # import from uois
        from src.util.utilities import subplotter, torch_to_numpy

        # visualization for uois
        rgb_imgs = torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)
        num_objs = np.unique(seg_mask).max() + 1
        rgb = rgb_imgs[0].astype(np.uint8)
        depth = batch["xyz"][0, 2].detach().cpu().numpy()
        seg_mask_plot = get_color_mask(seg_mask, nc=num_objs)
        center_offset = (center_offset - center_offset.min()) / (
            center_offset.max() - center_offset.min()
        )
        images = [
            rgb,
            depth,
            center_offset,
            initial_mask,
            get_color_mask(fg_mask),
            seg_mask_plot,
        ]
        titles = [
            "Image",
            "depth",
            "CenterOffset",
            "mask before rrn",
            "floor&table",
            "Refined Masks. #objects: {}".format(np.unique(seg_mask).shape[0] - 1),
        ]

        import matplotlib.pyplot as plt

        if return_ax:
            overlay = seg_mask_plot * 0.5 + rgb
            overlay = overlay / overlay.max() * 255
            plt.subplot(2, 2, 3)
            plt.axis("off")
            plt.imshow(overlay.astype(np.uint8))
            plt.title(self.__class__.__name__.lower())
        else:
            fig = subplotter(images, titles, fig_num=1)
            plt.show()
            plt.close()


# TODO
# PointNet2 Scan-net scene segmentation
class PN2(object):
    def __init__(self, **kwargs):
        super().__init__()

    def get_seg(self, rgb, depth):
        pass


class UCN(CategoryAgnosticSeg):
    def __init__(self, **kwargs):
        super().__init__()
        base_path = UCN_PATH

        # insert UCN path. (don't change the global env, there's confliction in package name)
        sys.path.insert(0, os.path.join(base_path, "lib"))
        sys.path.insert(0, os.path.join(base_path, "lib", "fcn"))
        from fcn.config import cfg, cfg_from_file
        from networks.SEG import seg_resnet34_8s_embedding

        cfg_from_file(UCN_CONFIG)
        cfg.device = DEVICE

        cfg.instance_id = 0
        num_classes = 2
        cfg.MODE = "TEST"
        cfg.TEST.VISUALIZE = False

        network_data = torch.load(UCN_CKPT_PATH1, map_location=DEVICE)
        self.network = seg_resnet34_8s_embedding(
            num_classes, cfg.TRAIN.NUM_UNITS, network_data
        ).to(device=cfg.device)
        self.network.eval()

        network_data_crop = torch.load(UCN_CKPT_PATH2, map_location=DEVICE)
        self.network_crop = seg_resnet34_8s_embedding(
            num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop
        ).to(device=cfg.device)
        self.network_crop.eval()

        sys.path.pop()
        sys.path.pop()

        # self.table_segmenter = UOIS() #PlaneDetector() # TODO tmp hack.

    def seg_fn(self, x):
        from test_dataset import test_sample

        return test_sample(x, self.network, self.network_crop)

    def seg_fn_dropout(self, x, p):
        from test_dataset import test_sample

        self.network.fcn.resnet34_8s.dropout = True
        self.network.fcn.resnet34_8s.dropout_p = p
        self.network_crop.fcn.resnet34_8s.dropout = True
        self.network_crop.fcn.resnet34_8s.dropout_p = p
        return test_sample(x, self.network, self.network_crop)

    # TODO(curtisa): Fix
    def get_segs(
        self,
        rgb_image,
        debug=DEFAULT_DEBUG,
        point_cloud=None,
        verbose=True,
        return_int=False,
        conservative=False,
        relabel_fg=False,
        dropout=False,
        dropout_p=0.5,
        num_segs=10,
        **kwargs
    ):

        # NOTE. ori input and output - 1) y axis pointing downward. 2) 0 for bg, 1+ for fg
        image_standardized = np.zeros_like(rgb_image).astype(np.float32)

        im_h, im_w, _ = rgb_image.shape
        for i in range(3):
            image_standardized[..., i] = (
                rgb_image[..., i] / 255.0 - self.image_net_mean[i]
            ) / self.image_net_std[i]
        image_standardized = image_standardized[..., ::-1].copy()

        # im in bgr order
        point_cloud[..., 1] *= -1  # y axis pointing downward. (reversed in UOIS)
        batch = {
            "image_color": torch.from_numpy(image_standardized)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .repeat(num_segs, 1, 1, 1),
            "depth": torch.from_numpy(point_cloud)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .repeat(num_segs, 1, 1, 1),
        }

        if dropout:
            _, instance_masks = self.seg_fn_dropout(
                batch, dropout_p
            )  # i=0 is background, objects start at i=1
        else:
            _, instance_masks = self.seg_fn(
                batch
            )  # i=0 is background, objects start at i=1

        segments = []
        instance_masks = instance_masks.detach().cpu().numpy()
        for i in range(instance_masks.shape[0]):
            instance_masks[i, :, :][instance_masks[i, :, :] == 1] = (
                instance_masks[i, :, :].max() + 1
            )

            if conservative:
                instance_masks[i, :, :] = self.erode(instance_masks[i, :, :])

            instances = np.unique(instance_masks[i, :, :])
            instances = instances[~np.isin(instances, BACKGROUND)]
            segment = np.zeros(rgb_image.shape[:2] + (2,), dtype=int)  # H x W x 2

            for j in instances:
                segment[instance_masks[i, :, :] == j, 0] = len(
                    YCB_CLASSES
                )  # unknown class
                segment[instance_masks[i, :, :] == j, 1] = j

            if not return_int:
                segment = str_from_int_seg(segment)

            segments.append(segment)

        return segments

    def get_seg(
        self,
        rgb_image,
        debug=DEFAULT_DEBUG,
        point_cloud=None,
        verbose=True,
        return_int=False,
        conservative=False,
        relabel_fg=False,
        dropout=False,
        dropout_p=0.5,
        **kwargs
    ):

        # NOTE. ori input and output - 1) y axis pointing downward. 2) 0 for bg, 1+ for fg
        image_standardized = np.zeros_like(rgb_image).astype(np.float32)
        im_h, im_w, _ = rgb_image.shape
        for i in range(3):
            image_standardized[..., i] = (
                rgb_image[..., i] / 255.0 - self.image_net_mean[i]
            ) / self.image_net_std[i]
        image_standardized = image_standardized[..., ::-1].copy()
        # im in bgr order
        point_cloud[..., 1] *= -1  # y axis pointing downward. (reversed in UOIS)
        batch = {
            "image_color": torch.from_numpy(image_standardized)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float(),
            "depth": torch.from_numpy(point_cloud)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float(),
        }

        if dropout:
            _, instance_mask = self.seg_fn_dropout(
                batch, dropout_p
            )  # i=0 is background, objects start at i=1
        else:
            _, instance_mask = self.seg_fn(
                batch
            )  # i=0 is background, objects start at i=1

        if instance_mask is not None:
            instance_mask = instance_mask[0].detach().cpu().numpy()
        else:
            instance_mask = np.zeros(rgb_image.shape[:2])
        instance_mask[instance_mask == 1] = instance_mask.max() + 1

        instances = np.unique(instance_mask)
        instances = instances[~np.isin(instances, BACKGROUND)]
        segment = np.zeros(rgb_image.shape[:2] + (2,), dtype=int)  # H x W x 2

        segment[np.isin(instance_mask, BACKGROUND), 0] = FLOOR
        for i in instances:
            segment[instance_mask == i, 0] = len(YCB_CLASSES)  # unknown class
            segment[instance_mask == i, 1] = i

        if not return_int:
            segment = str_from_int_seg(segment)

        return segment


class UOIS(CategoryAgnosticSeg):
    def __init__(self, base_path=None, ckpt_path=None, **kwargs):
        super().__init__()
        base_path = base_path if base_path is not None else UOIS_PATH
        sys.path.append(base_path)
        import src.segmentation as uois_segmentation

        ckpt_path = ckpt_path if ckpt_path is not None else UOIS_CKPT_PATH
        dsn_filename = os.path.join(
            ckpt_path, "DepthSeedingNetwork_3D_TOD_checkpoint.pth"
        )
        rrn_filename = os.path.join(ckpt_path, "RRN_OID_checkpoint.pth")
        UOIS3D_CONFIG["final_close_morphology"] = "TableTop_v5" in rrn_filename
        self.predictor = uois_segmentation.UOISNet3D(
            UOIS3D_CONFIG, dsn_filename, DSN_CONFIG, rrn_filename, RRN_CONFIG
        )  # , device=DEVICE)

        self.im_xyz_feat = None

        def feat_hook(module, inp, output):
            self.im_xyz_feat = (
                output.detach().cpu().numpy()
            )  # output = (scores, proposal_deltas)

        self.predictor.rrn.decoder.register_forward_hook(feat_hook)

    def relabel_fg(self, batch):
        # relabel holes in depth image as foreground(obj)
        # NOTE: might lead to oversegmention in relabeled region due to discontinuity in depth
        # TODO fill in holes of depth image(interpolate)
        from scipy.ndimage import binary_fill_holes

        fg_masks, _, _, _ = self.predictor.run_on_batch(batch)

        fg_mask = fg_masks[0].cpu().numpy()  # 0 is background, 1 is table, 2 is object
        table_masks_all = binary_fill_holes(fg_mask).astype(int)
        relabel_fg_pixels = np.logical_xor(table_masks_all, fg_mask >= 1)
        fg_mask[relabel_fg_pixels == 1] = 2
        return fg_mask[np.newaxis, ...]

    def get_seg(
        self,
        rgb_image,
        debug=DEFAULT_DEBUG,
        point_cloud=None,
        verbose=True,
        return_int=False,
        conservative=False,
        relabel_fg=False,
        **kwargs
    ):
        im_w = rgb_image.shape[1]
        ceil_h = np.ceil(rgb_image.shape[0] // 16 / 2.0) * 32
        padding_h = (ceil_h - rgb_image.shape[0]) // 2
        padding_h = 0 if padding_h < 0 else int(padding_h)
        if padding_h > 0:
            rgb_image = np.concatenate(
                (
                    np.zeros((padding_h, im_w, 3)).astype(np.uint8),
                    rgb_image,
                    np.zeros((padding_h, im_w, 3)).astype(np.uint8),
                ),
                axis=0,
            )
            point_cloud = np.concatenate(
                (
                    np.zeros((padding_h, im_w, 3)),
                    point_cloud,
                    np.zeros((padding_h, im_w, 3)),
                ),
                axis=0,
            )

        image_standardized = np.zeros_like(rgb_image).astype(np.float32)
        for i in range(3):
            image_standardized[..., i] = (
                rgb_image[..., i] / 255.0 - self.image_net_mean[i]
            ) / self.image_net_std[i]

        batch = {
            "rgb": torch.from_numpy(image_standardized)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float(),
            "xyz": torch.from_numpy(point_cloud)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float(),
        }
        if relabel_fg:
            fg_masks_relabeled = self.relabel_fg(batch)
            (
                fg_masks,
                center_offsets,
                initial_masks,
                seg_masks,
            ) = self.predictor.run_on_batch(
                batch, use_gt_fg=True, gt_fg=fg_masks_relabeled
            )
        else:
            (
                fg_masks,
                center_offsets,
                initial_masks,
                seg_masks,
            ) = self.predictor.run_on_batch(batch)

        fg_mask = fg_masks[0].cpu().numpy()  # 0 is background, 1 is table, 2 is object
        instance_mask = seg_masks[0].cpu().numpy()
        # fg_mask[(instance_mask==FLOOR) & (fg_mask==2)] = 0 # TODO: why does this occur and should it be bg or table?

        if conservative:
            instance_mask = self.erode(instance_mask)

        for i, cls in enumerate(BACKGROUND):
            instance_mask[fg_mask == cls] = i
        instances = np.unique(instance_mask)
        # instances = instances[~np.isin(instances, BACKGROUND)] # skip background and table

        segment = np.zeros(rgb_image.shape[:2] + (2,), dtype=int)  # H x W x 2
        segment[..., 0] = fg_mask
        segment[fg_mask == 2, 0] = len(YCB_CLASSES)  # unknown class

        for i in instances:  # i=0 is background, objects start at i=2
            segment[instance_mask == i, 1] = i
        segment[fg_mask == 1, 1] = 1  # 1 is table

        # either filter here or turn on open3d/trimesh noise removal
        segment[np.logical_xor(segment[..., 1] >= 2, fg_mask == 2), 0] = FLOOR  # table.

        if not return_int:
            segment = str_from_int_seg(segment)

        if debug:
            # # visualization
            self.vis_plot(rgb_image, instance_mask, **kwargs)

            # visualization for debugging purpose(including fg, center offest, etc)
            # self.vis(seg_masks[0].cpu().numpy(), fg_mask,
            #          center_offsets[0].cpu().numpy().transpose(1,2,0), initial_masks[0].cpu().numpy(), batch, **kwargs)
        return segment[padding_h:-padding_h] if padding_h > 0 else segment


class MultiSeg(object):
    def __init__(self, base_path=None, ckpt_path=None, post_classifier=False, **kwargs):
        self.any_cls_mask = UOIS(base_path, ckpt_path)  # TODO separate these paths
        if post_classifier:
            self.select_cls_det = FasterRCNN()
        else:
            self.select_cls_mask = MaskRCNN(**kwargs)
        self.use_classifier = post_classifier

    def erode(self, masks, kernel_size=3):
        from scipy.ndimage import binary_erosion

        for i in np.unique(masks[..., 1]):
            if i in BACKGROUND:
                continue
            mask = masks[..., 1] == i
            boundary_mask = np.logical_xor(
                binary_erosion(mask, structure=np.ones([kernel_size, kernel_size])),
                mask,
            )
            masks[boundary_mask] = (0, 0)
        return masks

    def get_cls(self, seg_im, instance_id, is_mask=False):
        if is_mask:
            return Counter(seg_im[instance_id, 0]).most_common(1)[0][0]
        return Counter(seg_im[seg_im[..., 1] == instance_id, 0]).most_common(1)[0][0]

    def get_seg(
        self,
        rgb,
        point_cloud=None,
        return_int=False,
        debug=DEFAULT_DEBUG,
        return_ax=True,
        conservative=True,
        **kwargs
    ):
        if self.use_classifier:  # class agnostic segmentation + detector
            cls_detect = self.select_cls_det.get_bbox(
                rgb, return_int=True, debug=debug, **kwargs
            )
            seg_any_cls = self.any_cls_mask.get_seg(
                rgb,
                point_cloud=point_cloud,
                return_int=True,
                debug=debug,
                return_ax=return_ax,
                **kwargs
            )
            seg_final = self.label_seg(cls_detect, seg_any_cls)
        else:  # use two segmentation network
            seg_select_cls_mask = self.select_cls_mask.get_seg(
                rgb, return_int=True, debug=debug, return_ax=return_ax, **kwargs
            )
            seg_any_cls = self.any_cls_mask.get_seg(
                rgb,
                point_cloud=point_cloud,
                return_int=True,
                debug=debug,
                return_ax=return_ax,
                **kwargs
            )
            seg_final = self.merge_seg(seg_select_cls_mask, seg_any_cls)

        if conservative:
            seg_final = self.erode(seg_final)

        if not return_int:
            seg_final = str_from_int_seg_general(
                seg_final, use_classifer=self.use_classifier
            )
        if debug:
            self.vis(seg_final.copy(), rgb)
        return seg_final

    ##################################################

    def get_iou(self, im1, id1, im2, id2):
        mask1 = (im1 == id1).astype(np.uint8)
        mask2 = (im2 == id2).astype(np.uint8)
        return np.logical_and(mask1, mask2).sum() / np.logical_or(mask1, mask2).sum()

    def get_iou_matrix(self, im1, idlist1, im2, idlist2):
        iou_matrix = np.zeros((len(idlist1), len(idlist2)))
        for i, obj_id_i in enumerate(idlist1):
            for j, obj_id_j in enumerate(idlist2):
                iou_matrix[i, j] = self.get_iou(im1, obj_id_i, im2, obj_id_j)
        return iou_matrix

    def merge_seg(
        self, seg_sel_cls, seg_any_cls, merge_threshold=0.75, noise_threshold=500
    ):
        selcls_obj_ids = np.unique(seg_sel_cls[..., 1])
        allcls_obj_ids = np.unique(seg_any_cls[..., 1])
        if len(selcls_obj_ids) == 0:
            return seg_any_cls.copy()
        if len(allcls_obj_ids) == 0:
            return seg_sel_cls.copy()

        # merged based on iou
        iou_matrix = self.get_iou_matrix(
            seg_sel_cls[..., 1], selcls_obj_ids, seg_any_cls[..., 1], allcls_obj_ids
        )

        segment_merge = seg_any_cls.copy()
        obj_ids = np.unique(segment_merge[..., 1])
        max_obj_id = obj_ids.max(initial=-1) + 1
        for i, obj_id in enumerate(obj_ids):
            # use uois mask + maskrcnn cls label if iou > threshold
            mask = seg_any_cls[..., 1] == obj_id
            if iou_matrix[..., i].max() < merge_threshold:
                # no matched maskrcnn object. cls unknown
                continue
            id_selected = np.argmax(iou_matrix[..., i])
            segment_merge[mask, 0] = self.get_cls(
                seg_sel_cls, selcls_obj_ids[id_selected]
            )
            selcls_obj_ids[id_selected] = -1

        for obj_id in selcls_obj_ids:
            mask = seg_sel_cls[..., 1] == obj_id
            if (
                (obj_id == -1)
                or (self.get_cls(seg_sel_cls, obj_id) in BACKGROUND)
                or (self.get_cls(seg_any_cls, mask, is_mask=True) in BACKGROUND)
                or (mask.sum() < noise_threshold)
            ):
                # TODO filter out as noise or assign to closest
                continue  # skip merged obj and table (use table from UOIS)
            segment_merge[mask, 0] = self.get_cls(seg_sel_cls, obj_id)
            segment_merge[mask, 1] = obj_id + max_obj_id
        return segment_merge

    ##################################################

    def get_iom(self, im1, id1, im2, id2):
        mask1 = (im1 == id1).astype(np.uint8)
        mask2 = (im2 == id2).astype(np.uint8)
        return np.logical_and(mask1, mask2).sum() / min(mask1.sum(), mask2.sum())

    def get_iom_matrix(self, im1, idlist1, im2, idlist2):
        iom_matrix = np.zeros((len(idlist1), len(idlist2)))
        for i, obj_id_i in enumerate(idlist1):
            for j, obj_id_j in enumerate(idlist2):
                iom_matrix[i, j] = self.get_iom(im1, obj_id_i, im2, obj_id_j)
        return iom_matrix

    def label_seg(self, seg_sel_cls, seg_any_cls, merge_threshold=0.75):
        selcls_obj_ids = np.unique(seg_sel_cls[..., 1])
        allcls_obj_ids = np.unique(seg_any_cls[..., 1])
        if len(selcls_obj_ids) == 0:
            return seg_any_cls.copy()
        if len(allcls_obj_ids) == 0:
            return seg_sel_cls.copy()

        # merged based on iom(intersection over min)
        iom_matrix = self.get_iom_matrix(
            seg_sel_cls[..., 1], selcls_obj_ids, seg_any_cls[..., 1], allcls_obj_ids
        )

        segment_merge = seg_any_cls.copy()
        obj_ids = np.unique(segment_merge[..., 1])
        for i, obj_id in enumerate(obj_ids):
            # use uois mask + maskrcnn cls label if iou > threshold
            mask = seg_any_cls[..., 1] == obj_id
            if iom_matrix[..., i].max() < merge_threshold:
                # no matched bbox. cls unknown
                continue
            id_selected = np.argmax(iom_matrix[..., i])
            cls = self.get_cls(seg_sel_cls, selcls_obj_ids[id_selected])
            if FASTERRCNN_CLASS[cls] in [BOWL, CUP]:
                segment_merge[mask, 0] = cls
        return segment_merge

    ##################################################

    def overlay_pred(self, segim, rgb_image):
        from detectron2.data import MetadataCatalog
        from detectron2.structures import Instances
        from detectron2.utils.visualizer import ColorMode, Visualizer

        pred_objs = np.unique(segim[..., 1])
        MetadataCatalog.get("tmp").set(
            thing_classes=[self.get_cls(segim, obj_id) for obj_id in pred_objs]
        )
        sync_metadata = MetadataCatalog.get("tmp")
        v = Visualizer(
            rgb_image,
            metadata=sync_metadata,
            scale=1.0,
            instance_mode=ColorMode.SEGMENTATION,
        )
        result_wrap = Instances(
            rgb_image.shape[:2],
            pred_masks=torch.BoolTensor(
                [segim[..., 1] == obj_id for obj_id in pred_objs]
            ),
            pred_classes=np.arange(len(pred_objs)),
        )
        v = v.draw_instance_predictions(result_wrap)
        MetadataCatalog.remove("tmp")
        return v.get_image()

    def vis(self, merge, rgb_image):
        import matplotlib.pyplot as plt

        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.imshow(self.overlay_pred(merge, rgb_image))
        plt.title(self.__class__.__name__.lower())
        # plt.show()
        # plt.close()


##################################################


def init_seg(base_path=None, ckpt_path=None, branch="maskrcnn", **kwargs):
    """
    initialize segmentation network . maskrcnn | uois | all

    base_path: root dir to vision repositories
    ckpt_path: path to pretrained model (.pth)

    """
    if branch == "maskrcnn":
        return MaskRCNN(**kwargs)
        # return FasterRCNN()
    elif branch == "uois":
        return UOIS(base_path, ckpt_path)
    elif branch == "ucn":
        return UCN(**kwargs)
    elif branch == "all":
        return MultiSeg(base_path, ckpt_path, **kwargs)
    raise NotImplementedError(branch)
