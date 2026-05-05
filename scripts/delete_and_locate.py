import os
import ast
import configparser
import shutil
from tqdm import tqdm
from pypcd4 import PointCloud
import numpy as np
from bev_detector import BEV_DETECTOR
from ultralytics import YOLO
import json


config = configparser.ConfigParser()
config.read('settings.conf')


def main():
    model = YOLO('runs/bev_obb_model9/weights/best.engine', task='obb')
    detector = BEV_DETECTOR(model)

    fold_num = "001"
    pcd_num = "07"
    hd_map_path = f"training/HD_my/{fold_num}.pcd"
    current_pcd_path = f"training/PCD/{fold_num}/{pcd_num}.pcd"
    pcd_folder_path = f"training/PCD/{fold_num}/poses.json"
    
    with open(pcd_folder_path) as f:
        json_poses = json.load(f)
        position = json_poses[int(pcd_num)]['position']
    
    clear_points = detector.detect_and_delete(current_pcd_path, position['x'], position['y'])
    




if __name__ == "__main__":
    main()
