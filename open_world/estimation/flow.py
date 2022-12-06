import cv2
import numpy as np
import torch


# https://github.com/zhengqili/Neural-Scene-Flow-Fields
def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    )

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return fwd_mask, bwd_mask


# https://github.com/princeton-vl/RAFT
class OpticalFlow(object):
    def __init__(
        self,
        base_path="/home/xiaolinf/project/perception/tracking/RAFT",
        device=None,
        bid=True,
    ):
        import argparse
        import sys

        sys.path.append(base_path)
        sys.path.append(base_path + "/core")
        from raft import RAFT
        from raftutils import flow_viz
        from raftutils.utils import InputPadder

        model = torch.nn.DataParallel(
            RAFT(
                argparse.Namespace(
                    small=False, mixed_precision=False, alternate_corr=False
                )
            )
        )
        model.load_state_dict(torch.load(base_path + "/models/raft-things.pth"))
        model = model.module
        if device is None:
            device = torch.device("cuda")
        model.to(device)
        self.device = device
        model.eval()
        self.model = model
        self.inputpadder = InputPadder
        self.flow_viz = flow_viz
        self.bidirection = bid

    def forward(self, image1, image2):
        # input - im_1 b x h x w x 3. 0-255. tensor
        # output - cpu. numpy. hxwx2
        assert image1.max() > 125
        image1 = image1.permute(0, 3, 1, 2).float().to(self.device)
        image2 = image2.permute(0, 3, 1, 2).float().to(self.device)

        padder = self.inputpadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            if self.bidirection:
                flow_fwd = (
                    self.model(image1, image2, iters=20, test_mode=True)[1]
                    .detach()
                    .cpu()
                    .numpy()[0]
                    .transpose(1, 2, 0)
                )
                flow_bwd = (
                    self.model(image2, image1, iters=20, test_mode=True)[1]
                    .detach()
                    .cpu()
                    .numpy()[0]
                    .transpose(1, 2, 0)
                )
                fwd_mask, bwd_mask = compute_fwdbwd_mask(flow_fwd, flow_bwd)
                flow = flow_fwd * fwd_mask[..., None]
                return flow
            else:
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                return flow_up.detach().cpu().numpy()[0].transpose(1, 2, 0)

    def flow_to_im(self, flo):
        # HW3. flow as pred+concat zero torchtensor
        flo = self.flow_viz.flow_to_image(flo[..., :2]).astype(np.uint8)  # hw3
        # img_flo = np.concatenate([img, flo], axis=0)
        return flo


# class FarneBackFlow(object):
#     def __init__(self, bid=False):
#         self.bidirection = bid

#     def forward(self, im_1, im_2):
#         # 1HW3, 0-255. tensor
#         assert im_1.shape[0]==1
#         assert im_2.shape[0]==1
#         im_1 = im_1[0].numpy().astype(np.uint8)
#         im_2 = im_2[0].numpy().astype(np.uint8)
#         assert im_1.max()>125
#         im_1_gray = cv2.cvtColor(im_1,cv2.COLOR_RGB2GRAY)
#         im_2_gray = cv2.cvtColor(im_2,cv2.COLOR_RGB2GRAY)
#         if self.bidirection:
#             flow_fwd = cv2.calcOpticalFlowFarneback(im_1_gray,im_2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#             flow_bwd = cv2.calcOpticalFlowFarneback(im_2_gray,im_1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#             fwd_mask,bwd_mask = compute_fwdbwd_mask(flow_fwd, flow_bwd)
#             flow = flow_fwd * fwd_mask[...,None]
#         else:
#             flow = cv2.calcOpticalFlowFarneback(im_1_gray,im_2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         return flow

#     def flow_to_im(self, scene_flow_vis_bs0):
#         # HW3. torchtensor
#         hsv = np.zeros_like(scene_flow_vis_bs0)
#         hsv[...,1] = 255
#         mag, ang = cv2.cartToPolar(scene_flow_vis_bs0[...,0], scene_flow_vis_bs0[...,1])
#         # print(
#         hsv[...,0] = ang*180/np.pi/2
#         hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#         flow_im = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)
#         return flow_im
