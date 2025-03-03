import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from torch.autograd import Function
from eval.emd_pytorch_implement.pkg.emd_loss_layer import EMDLoss


def get_emd_loss(pred, gt):    
    dist = EMDLoss()
    cost = dist(pred, gt)
    loss = (torch.sum(cost))/(pred.size()[1]*gt.size()[0])

    return loss


def pairwise_distance(x, y):
    # Compute pairwise distance between each point in x and each point in y
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    xy = torch.bmm(x, y.permute(0, 2, 1))
    dist = xx - 2 * xy + yy.permute(0, 2, 1)
    return dist

def chamfer_distance(x, y):
    # Compute Chamfer Distance between two point clouds x and y
    dist = pairwise_distance(x, y)
    dist1 = torch.min(dist, dim=2)[0]
    dist2 = torch.min(dist, dim=1)[0]
    return dist1, dist2

class DensityAwareChamferDistance(nn.Module):
    def __init__(self, alpha=1000, n_lambda=1.0):
        super(DensityAwareChamferDistance, self).__init__()
        self.alpha = alpha
        self.n_lambda = n_lambda

    def forward(self, x, y):
        # x, y are point clouds of shape (batch_size, num_points, 3)
        B, N, _ = x.size()
        _, M, _ = y.size()

        # Calculate basic Chamfer distances
        dist1, dist2 = chamfer_distance(x, y)
        exp_dist1, exp_dist2 = torch.exp(-dist1 * self.alpha), torch.exp(-dist2 * self.alpha)

        # Calculate density weights
        idx1 = torch.argmin(pairwise_distance(x, y), dim=2) 
        idx2 = torch.argmin(pairwise_distance(y, x), dim=2)

        count1 = torch.zeros(B, M).to(x.device).scatter_add_(1, idx1, torch.ones(B, N).to(x.device))  
        count2 = torch.zeros(B, N).to(x.device).scatter_add_(1, idx2, torch.ones(B, M).to(x.device))  

        weight1 = count1.gather(1, idx1).float().detach() ** self.n_lambda    
        weight2 = count2.gather(1, idx2).float().detach() ** self.n_lambda     

        weight1 = (weight1 + 1e-6) ** (-1)
        weight2 = (weight2 + 1e-6) ** (-1)

        loss1 = (1 - exp_dist1 * weight1).mean(dim=1)
        loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

        # Compute weighted Chamfer distance
        loss = (loss1 + loss2) / 2

        return loss.mean()


def knn(x, c, k):
    """
    Input:
        x: [B, N, C]
        c: [B, G, C]
        int: k
    Return:
        idx: [B, G, M]
    """
    pairwise_distance = torch.cdist(c, x, p=2) 
 
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  

    return idx


def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)    
    distance = torch.ones(B, N).to(device) * 1e10                       
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  
    batch_indices = torch.arange(B, dtype=torch.long).to(device)    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    pcentroids = xyz[batch_indices[:, None], centroids, :]
    return pcentroids


def chamfer_distance2(point_cloud1, point_cloud2):
    B, N1, _ = point_cloud1.shape
    _, N2, _ = point_cloud2.shape
    dist1 = torch.sum(torch.min(torch.norm(point_cloud1.unsqueeze(2) - point_cloud2.unsqueeze(1), dim=-1), dim=-1)[0]) 
    dist2 = torch.sum(torch.min(torch.norm(point_cloud2.unsqueeze(2) - point_cloud1.unsqueeze(1), dim=-1), dim=-1)[0])
    CD2 = dist1/N1 + dist2/N2
    return CD2/B

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist = dist + torch.sum(src ** 2, -1).view(B, N, 1)
    dist = dist + torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist 

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=True)
    return group_idx

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def get_repulsion_loss(pred, nsample=20, h=0.7):
    ''' 
    pred : torch.tensor[batch_size, npoint, 3]
    '''
    assert(pred.shape[1] >= nsample), 'rep loss : point number is less than nsample'
    fps_idx = farthest_point_sample(pred, pred.shape[1])
    new_xyz = index_points(pred, fps_idx)
    idx = knn_point(nsample, pred, new_xyz)
    grouped_pred = index_points(pred, idx)     
    grouped_pred -= pred.unsqueeze(2)       


    dist_square = torch.sum(grouped_pred ** 2, dim=-1)     
    assert(dist_square.shape[2] >= 6), 'rep loss : group point number is less than k'
    dist_square, idx = torch.topk(dist_square, k=6, dim=-1, largest=False, sorted=True)   
    dist_square = dist_square[:, :, 1:]  

    dist = torch.sqrt(dist_square) 
    loss = F.relu(h-dist)     
    uniform_loss = torch.mean(loss)
    return uniform_loss


