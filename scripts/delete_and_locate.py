import os
import ast
import configparser
import shutil
import numpy as np
from tqdm import tqdm
from pypcd4 import PointCloud
from bev_detector import BEV_DETECTOR
from localizer import Localizer
from ultralytics import YOLO
import json


config = configparser.ConfigParser()
config.read('settings.conf')


def main():
    model = YOLO('runs/bev_obb_model9/weights/best.engine', task='obb')
    detector = BEV_DETECTOR(model)

    fold_num = "001"
    pcd_num = "17"
    hd_map_path = f"training/HD_my/{fold_num}.pcd"
    current_pcd_path = f"training/PCD/{fold_num}/{pcd_num}.pcd"
    poses_folder_path = f"training/PCD/{fold_num}/poses.json"
    
    hd_map = PointCloud.from_path(hd_map_path)
    hd_map = hd_map.numpy(("x", "y", "z"))
    
    scan_localizer = Localizer(hd_map)
    
    with open(poses_folder_path) as f:
        json_poses = json.load(f)
        position = json_poses[int(pcd_num)]['position']
        
    clear_points = detector.detect_and_delete(current_pcd_path, position['x'], position['y'])
    
    
    clear_points[:, 0] -= position['x']
    clear_points[:, 1] -= position['y']
    clear_points[:, 2] -= position['z']
    
    transform = scan_localizer.process_frame(clear_points)
    print(transform)
    print(position)




if __name__ == "__main__":
    main()
