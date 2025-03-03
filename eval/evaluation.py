import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                     
import torch
import torch.nn as nn
import time
import config
from collections import defaultdict
import checkpoints
import numpy as np
import evaltools
from tqdm import tqdm
from glob import glob
from tools import utils
import random

# Dataloader
data_folder = '/data/path/to/ShapeNet_test1024/'
real_folder = '/data/path/to/ShapeNet_test4096/'                     


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(2024)


input_modelist = sorted(os.listdir(data_folder))
real_modelist = sorted(os.listdir(real_folder))

assert len(input_modelist) == len(real_modelist), 'Dataset NUM Error!'

lenth = 8                                                                  

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")
cfg = config.load_config('../cfgs/setting.yaml')

model = config.get_model(cfg)
model.to(device)
model = nn.DataParallel(model).cuda()
checkpoint_io = checkpoints.CheckpointIO('../out', model=model)
load_dict = checkpoint_io.load('model_best.pt')

model.eval() 

eva_1 = 0 
eva_2 = 0
eva_3 = 0
eva_4 = 0
num = len(input_modelist)
start = time.time()
for idx in tqdm(range(num)):

    points_in = np.loadtxt(os.path.join(data_folder, input_modelist[idx]))
    gt_points_in = np.loadtxt(os.path.join(real_folder, real_modelist[idx]))

    bbox=np.zeros((2,3))
    bbox[0][0]=np.min(points_in[:,0])
    bbox[0][1]=np.min(points_in[:,1])
    bbox[0][2]=np.min(points_in[:,2])
    bbox[1][0]=np.max(points_in[:,0])
    bbox[1][1]=np.max(points_in[:,1])
    bbox[1][2]=np.max(points_in[:,2])
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    scale1 = 1/scale
    for i in range(points_in.shape[0]):
        points_in[i]=points_in[i]-loc
        points_in[i]=points_in[i]*scale1
    #upsampling
    points = torch.tensor(points_in).to(device).float().unsqueeze(0)
    gt_points = torch.tensor(gt_points_in).to(device).float()
    
    for idxy in range(lenth):
        set_seed(idxy)
        pts_in = points
        with torch.no_grad():
            completion_points = model(pts_in, device, idxy=idxy, vis=True)
        pts_ot = completion_points

        if(idxy==0):
            pts_gen = pts_ot     
        else:
            pts_gen = torch.cat([pts_gen, pts_ot], dim=1)        

    print('pts_gen:', pts_gen.shape)
    pts_up = pts_gen.squeeze(0)

    pairwise_distance = torch.cdist(pts_up, pts_up, p=2)
    dist, _ = pairwise_distance.topk(k=30, dim=-1, largest=False)
    avg=torch.mean(dist, axis=1)
    avgtotal=torch.mean(dist)
    fps_idx=torch.where(avg<avgtotal*2.0)[0]                                      
    pts_up=pts_up[fps_idx,:]
    print('pts_up:', pts_up.shape)

    loc = torch.tensor(loc).to(device).float()
    scale = torch.tensor(scale).to(device).float()
    up_points = pts_up
    for i in range(up_points.shape[0]):
        up_points[i]=up_points[i]*scale
        up_points[i]=up_points[i]+loc
    up_points = utils.fps(up_points.unsqueeze(0), 4096)                    
    gt_points = gt_points.unsqueeze(0)

    cd2 = evaltools.chamfer_distance2(up_points, gt_points)
    emd = evaltools.get_emd_loss(up_points, gt_points)
    hd = evaltools.hausdorff_distance(up_points, gt_points)
    fscore = evaltools.fscore(up_points, gt_points)

    eva_1 += cd2
    eva_2 += emd
    eva_3 += hd
    eva_4 += fscore
  
    #outs = os.path.join(out_uppt, input_modelist[idx])
    #np.savetxt(outs, up_points.squeeze(0).cpu().numpy())

eva_1 = eva_1 / num * 1000
eva_2 = eva_2 / num * 1000
eva_3 = eva_3 / num * 1000
eva_4 = eva_4 / num * 1000
print('cd2: ', eva_1)
print('emd: ', eva_2)
print('hd: ', eva_3)
print('fscore: ', eva_4)


end = time.time()
print('time cost:%.5f sec'%(end-start))






