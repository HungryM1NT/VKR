import os
import ast
import configparser
from pypcd4 import PointCloud
import numpy as np
from utils.general import pcd_from_points


config = configparser.ConfigParser()
config.read('scripts/settings.conf')

PCD_COL = ast.literal_eval(config['CONSTANTS']['PCD_COLUMNS'])

def voxel_down_sample(pcd_points, voxel_size):
    xyz = pcd_points[:, :3]
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    downsampled_points = pcd_points[unique_indices]
    
    return downsampled_points
    
def create_HD_map(pcds_folder_path, save_path):
    hd_map_list = []
    
    files = os.listdir(pcds_folder_path)
    
    for file in files:
        if file.endswith('.pcd'):
            pcd = PointCloud.from_path(f"{pcds_folder_path}/{file}")
            pcd_points_array = pcd.numpy(("x", "y", "z", "i"))
            points_array = voxel_down_sample(pcd_points_array, 0.1)
            hd_map_list.append(points_array)
    
    hd_map_points = np.vstack(hd_map_list)
    hd_map_points = voxel_down_sample(hd_map_points, 0.05)
    hd_map = pcd_from_points(hd_map_points)
    hd_map.save(save_path)