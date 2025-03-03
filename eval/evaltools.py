import os
import torch  
from scipy.spatial.distance import directed_hausdorff
from emd_pytorch_implement.pkg.emd_loss_layer import EMDLoss
import torch.nn.functional as F

def chamfer_distance2(point_cloud1, point_cloud2):
    B, N1, _ = point_cloud1.shape
    _, N2, _ = point_cloud2.shape
    dist1 = torch.sum(torch.min(torch.norm(point_cloud1.unsqueeze(2) - point_cloud2.unsqueeze(1), dim=-1), dim=-1)[0]) 
    dist2 = torch.sum(torch.min(torch.norm(point_cloud2.unsqueeze(2) - point_cloud1.unsqueeze(1), dim=-1), dim=-1)[0])
    CD2 = dist1/N1 + dist2/N2
    return CD2/B


def hausdorff_distance(point_set_A, point_set_B):  
    distances_A_to_B = torch.cdist(point_set_A, point_set_B)
    distances_B_to_A = torch.cdist(point_set_B, point_set_A)
      
    max_distances_A_to_B = torch.min(distances_A_to_B, dim=-1)[0]
    max_distances_B_to_A = torch.min(distances_B_to_A, dim=-1)[0]  
      
    hausdorff_distance = torch.max(max_distances_A_to_B.max(), max_distances_B_to_A.max())  
      
    return hausdorff_distance


def get_emd_loss(pred, gt):
    
    dist = EMDLoss()
    cost = dist(pred, gt)
    loss = (torch.sum(cost))/(pred.size()[1]*gt.size()[0])

    return loss

def fscore(pts1, pts2, threshold=0.1):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    dist1 = torch.cdist(pts1, pts2)  
    dist2 = torch.cdist(pts2, pts1)  
    precision_1 = torch.mean((dist1 < threshold).float())
    precision_2 = torch.mean((dist2 < threshold).float())
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore


def mean(x, gt):
    m = torch.cdist(x, gt)
    m = torch.min(m, dim=2)[0]
    m = torch.mean(m)

    return m

def std(x, gt):
    td = torch.cdist(x, gt)
    td = torch.min(td, dim=2)[0]
    td = torch.std(td)

    return td



