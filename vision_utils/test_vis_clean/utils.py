import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def outer(a, b):
    assert a.shape[0] == b.shape[0]
    M = a.shape[1]
    N = b.shape[1]
    # print(a.shape,b.shape)
    a = a.view(1, 4, 1).repeat(1, 1, N)
    b = b.view(1, 1, 4).repeat(1, M, 1)
    # print(a.shape,b.shape)
    # a = F.repeat(a[:, :, None], N, axis=2)
    # b = F.repeat(b[:, None, :], M, axis=1)
    return (a * b)[0]


class QuaternionMatrix(nn.Module):
    def __init__(self):
        super(QuaternionMatrix, self).__init__()

    def forward(self, q, t):
        R = torch.eye(4)
        # print(R.shape,q.shape)
        R[0, 0] = 1 - q[2, 2] - q[3, 3]
        R[0, 1] = q[1, 2] - q[3, 0]
        R[0, 2] = q[1, 3] + q[2, 0]
        R[1, 0] = q[1, 2] + q[3, 0]
        R[1, 1] = 1 - q[1, 1] - q[3, 3]
        R[1, 2] = q[2, 3] - q[1, 0]
        R[2, 0] = q[1, 3] - q[2, 0]
        R[2, 1] = q[2, 3] + q[1, 0]
        R[2, 2] = 1 - q[1, 1] - q[2, 2]
        R[0, 3] = t[0]
        R[1, 3] = t[1]
        R[2, 3] = t[2]
        return R


class ICC(nn.Module):
    def __init__(self, quat, tran, model):
        super(ICC, self).__init__()
        self.quat = torch.nn.Parameter(torch.Tensor(np.asarray(quat)))
        self.tran = torch.nn.Parameter(torch.Tensor(np.asarray(tran)))
        # self.model = model
        self.model = np.concatenate((model, np.ones((model.shape[0], 1))), 1).transpose(
            1, 0
        )
        self.model = torch.Tensor(self.model)
        # print(self.quat,self.tran)
        self.xmap = (
            np.array([[i for i in range(640)] for j in range(480)]) / 640.0 * 2 - 1.0
        )  #   ( py/480. - 0.5)*-2
        self.ymap = (
            np.array([[480 - j for i in range(640)] for j in range(480)]) / 480.0 * 2
            - 1.0
        )  # (px/640. - 0.5)*2
        self.ones = np.ones((1, 640 * 480))

    def quaternion_matrix(self, quaternion, tran):
        quaternion = quaternion.view(1, -1)  # .contiguous()
        norm = torch.sum(quaternion ** 2)
        quaternion = quaternion * torch.sqrt(2.0 / norm)
        quaternion = outer(quaternion, quaternion)
        matrix = QuaternionMatrix()(quaternion, tran)

        return matrix

    def forward(
        self,
        mask,
        depth,
        voxels,
        view_matrix,
        proj_matrix,
        ivip,
        centers,
        free,
        occupied_by_other,
        masks,
    ):
        """
        pred: N x 3
        depth: H x W x 3
        voxels: Voxelgrid
        """
        import time

        start = time.time()
        mat_eye = self.quaternion_matrix(self.quat, self.tran)
        # self.quat.register_hook(lambda grad:print(grad))
        # self.tran.register_hook(lambda grad:print(grad))
        # mat_eye.register_hook(lambda grad:print('mat eye', grad.max(),grad.min()))
        # print(mat_eye)
        # print(model.shape)
        mat_world = torch.matmul(torch.Tensor(np.linalg.inv(view_matrix)), mat_eye)
        # print(mat_world)
        pred = torch.matmul(mat_world, self.model)  # 4 x M
        pred = (pred / pred[-1])[:3, :].transpose(1, 0)  # .contiguous() # M x 3
        # pred = pred.detach().clone().requires_grad_(True)
        # pred.register_hook(lambda grad:print(f'pred grad : {grad}, {grad.max()}, {grad.min()}'))
        # print(pred.mean(0))
        # print(pred.shape)
        # pred = qua
        # lower = [np.min(pred[:,0]),np.min(pred[:,1]),np.min(pred[:,2])]
        # upper = [np.max(pred[:,0]),np.max(pred[:,1]),np.max(pred[:,2])]
        # get relavant voxels(mask surface + pred occupied)
        # py,px = np.where(mask==1)
        # depth_scaled = (depth-0.5)*2
        # normed_loc = np.concatenate((self.xmap.reshape(1,-1),self.ymap.reshape(1,-1),depth_scaled.reshape(1,-1),self.ones))
        # point3d = ivip.dot(normed_loc)
        # point3d = (point3d/point3d[-1])[:3,:].reshape(3,480,640).transpose(1,2,0)
        # point3d = torch.Tensor(point3d)
        # masked_voxel = []
        # for i in range(px.shape[0]):
        # masked_voxel.append(voxel.voxel_from_point(point3d[py[i],px[i]]))
        # masks=torch.zeros(len(voxels.voxels))#.cuda()
        # free=torch.zeros(len(voxels.voxels))#.cuda()
        # occupied_by_other=torch.zeros(len(voxels.voxels))#.cuda()
        occupied_by_pred = torch.zeros(len(voxels.voxels))  # .cuda()
        # print(f'{len(voxels.voxels)} voxels in total')
        # occupied_by_pred.register_hook(lambda grad:print(f'occupied by pred grad: {grad}'))
        #######################################################
        # centers = torch.Tensor([voxels.center_from_voxel(voxel) for voxel in voxels.voxels]) # V x 3
        pred2center = pred.view(1, -1, 3).repeat(centers.shape[0], 1, 1) - centers.view(
            -1, 1, 3
        ).repeat(
            1, pred.shape[0], 1
        )  # V x N x 3
        pred2center_voxel_coord = pred2center / voxels.resolution  # VxNx3
        distance_voxel_coord = torch.norm(pred2center_voxel_coord, p=2, dim=2)  # VxN
        distance_voxel_coord_min_, distance_voxel_coord_min_ind = torch.min(
            distance_voxel_coord, dim=1
        )  # V, V
        distance_voxel_coord_min = torch.clamp(
            distance_voxel_coord_min_, max=voxels.resolution / 2.0
        )  # V
        occupand = 1 - distance_voxel_coord_min / (voxels.resolution / 2.0)  # V
        # occupied_by_pred[i]=occupand

        # distance = torch.norm(point3d.view(1,480,640,3).repeat(centers.shape[0],1,1,1)-centers.view(-1,1,1,3).repeat(1,480,640,1),p=2,dim=3) # Vx480x640
        # # # print(distance.shape)
        # # # import pdb;pdb.set_trace()
        # least_distance,indx = torch.min(distance,dim=2) # Vx480, Vx480
        # least_distance,indy = torch.min(least_distance,dim=1) # V, V
        for i, voxel in enumerate(voxels.voxels):
            occupand = 1 - distance_voxel_coord_min[i] / (voxels.resolution / 2.0)
            if occupand > 0:
                occupied_by_pred[i] = occupand
            # if least_distance[i]>1.414*voxels.resolution/2.:
            #     free[i]=1
            # elif mask[indy[i],indx[i][indy[i]]]==1:
            #     masks[i]=1
            # else:
            #     occupied_by_other[i]=1
        ##########################################################
        # if least_distance>voxels.resolution/2.:
        #     free[i]=1
        # elif mask[indy,indx[indy]]==1:
        #     masks[i]=1
        # else:
        #     occupied_by_other[i]=1

        # for i,voxel in enumerate(voxels.voxels):
        #     # if i%100==0:print(i);
        #     center = torch.Tensor(voxels.center_from_voxel(voxel))
        #     pred2center = pred-center # N x 3
        #     # pred2center.register_hook(lambda grad:print(grad,grad.max(),grad.min()))
        #     pred2center_voxel_coord = pred2center/voxels.resolution # Nx3
        #     distance_voxel_coord = torch.norm(pred2center_voxel_coord, p=2,dim=1) #Nx1
        #     distance_voxel_coord_min_ = torch.min(distance_voxel_coord) # 1
        #     # print(distance_voxel_coord_min)
        #     distance_voxel_coord_min = torch.clamp(distance_voxel_coord_min_, max=voxels.resolution/2.)
        #     occupand = 1 - distance_voxel_coord_min/(voxels.resolution/2.)
        #     # occupand.register_hook(lambda grad:print('occupand', grad,grad.max(),grad.min()))
        #     # occupied_by_pred[i]=occupand
        #     if occupand>0: occupied_by_pred[i]=occupand;
        #     # import pdb;pdb.set_trace()

        #     distance = torch.norm(point3d-center,p=2,dim=2) # 480x640
        #     # print(distance.shape)
        #     # import pdb;pdb.set_trace()
        #     least_distance,indx = torch.min(distance,dim=1) # 480, 480
        #     least_distance,indy = torch.min(least_distance,dim=0) # 1, 1
        #     if least_distance>voxels.resolution/2.:
        #         free[i]=1
        #     elif mask[indy,indx[indy]]==1:
        #         masks[i]=1
        #     else:
        #         occupied_by_other[i]=1
        pred_sum = occupied_by_pred.sum()
        # if pred_sum==0: pred_sum=pred_sum+1;
        mask_sum = masks.sum()
        # print(free.sum(),masks.sum(),occupied_by_pred.sum())
        # print(pred_sum,mask_sum)
        loss = 0
        if pred_sum > 0:
            loss += ((free + occupied_by_other) * occupied_by_pred).sum() / pred_sum
        if mask_sum > 0:
            loss -= (masks * occupied_by_pred).sum() / mask_sum
        # import pdb;pdb.set_trace()
        print(loss)
        return loss
