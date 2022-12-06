## Vision packages User Guide

### Installation
Necesarry code are already included as git submodules. 

To install dependencies:
- Install detectron2 following the instructions [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).(optional)
    - This is optional and only needed for tasks that involve object categories.


### Pretrained Models
Pretrained network models are managed using [git-lfs](https://git-lfs.github.com/).

#### Intall Git-lfs
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh |bash
apt-get install git-lfs
git lfs install
```
Or follow the steps on [git-lfs](https://git-lfs.github.com/).

#### Pull pretrained models
```
git lfs pull
```

#### List of pretrained models

Segmentation:
- maskrcnn_res50_ycb_real+syn.pth(default) / maskrcnn_res101_ycb.pth / maskrcnn_res50_ycb_2.pth
    Use MaskRCNN to segment YCB objects on RGB images.
- maskrcnn_rgbd.pth
    Use MaskRCNN to segment YCB objects on RGB-D images.
- bowlcup_detector.pth
    Use FasterRCNN to detect bowl or cup on RGB images.
- DepthSeedingNetwork_3D_TOD_checkpoint.pth / RRN_OID_checkpoint.pth
    Use UOIS to segment table plane and tabletop objects on RGB-D images(cagetory-agnostic). 
    UOIS repo pretrained model.

Shape Completion:
- msn1.pth
    Use MSN to complete partial pointcloud.


### Reference list
* https://github.com/chrisdxie/uois
* https://github.com/NVlabs/UnseenObjectClustering
* https://github.com/facebookresearch/detectron2
* https://github.com/Learning-and-Intelligent-Systems/AtlasNet
    Adapted from
    * https://github.com/Colin97/MSN-Point-Cloud-Completion
    * https://github.com/ThibaultGROUEIX/AtlasNet (Deprecated)


