from torch.utils import data
import os
import numpy as np
import open3d as o3d


def load_xyz_file(file_path):
    point_cloud = o3d.io.read_point_cloud(file_path)
    points = np.asarray(point_cloud.points)
    
    return points


class Shaping(data.Dataset):

    def __init__(self, dataset_folder, split, cfg):
        # Get all models
        self.dataset_folder = dataset_folder   
        self.split = split
        self.cfg = cfg
        self.models = os.listdir(dataset_folder)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.dataset_folder, model)
        data = load_xyz_file(model_path)

        input, _ = rotate_point_cloud_and_gt(data)
        input, _, _ = random_scale_point_cloud_and_gt(input, scale_low=0.8, scale_high=1.2)

        return data

def nomc(input, gt=None):
    '''
        make input in a unit sphere
        input: B N C
        
        output: B N C
    '''
    # the center point of input
    input_centroid = np.mean(input, axis=1, keepdims=True)
    input = input - input_centroid
    # (b, 1)
    input_furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True)
    # normalize to a unit sphere
    input = input / np.expand_dims(input_furthest_distance, axis=-1)

    return input

def rotate_point_cloud_and_gt(input, gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    angles = np.random.uniform(size=(3)) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    input = np.dot(input, rotation_matrix)
    if gt is not None:
        gt = np.dot(gt, rotation_matrix)
    return input, gt


def random_scale_point_cloud_and_gt(input, gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(scale_low, scale_high)
    input = np.multiply(input, scale)
    if gt is not None:
        gt = np.multiply(gt, scale)
    return input, gt, scale

