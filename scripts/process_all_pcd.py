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
from utils.general import pcd_from_points


config = configparser.ConfigParser()
config.read('settings.conf')

PCD_COL = ast.literal_eval(config['CONSTANTS']['PCD_COLUMNS'])
PCD_PATH = config['PATHS']['PCD_PATH']
PCD_DELETED_PATH = config['PATHS']['PCD_DELETED_PATH']


def main():
    if not os.path.exists(PCD_DELETED_PATH):
        model = YOLO('runs/bev_obb_model9/weights/best.engine', task='obb')
        detector = BEV_DETECTOR(model)
        
        os.makedirs(PCD_DELETED_PATH, exist_ok=True)
        
        pcd_folders = os.listdir(PCD_PATH)
        
        for pcd_folder_num in tqdm(pcd_folders, desc="Pandaset folders", position=0):
            pcd_deleted_full_path = f"{PCD_DELETED_PATH}/{pcd_folder_num}"
            os.makedirs(pcd_deleted_full_path, exist_ok=True)

            pcd_folder_path = f'{PCD_PATH}/{pcd_folder_num}'
            
            pcd_files = os.listdir(pcd_folder_path)
            pcd_files.sort()
            
            with open(f"{pcd_folder_path}/poses.json") as f:
                json_poses = json.load(f)
                
            for pcd_file in pcd_files:
                if pcd_file.endswith('.pcd'):
                    pcd_path = f"{pcd_folder_path}/{pcd_file}"
                    position = json_poses[int(pcd_file[:-4])]['position']
                    clear_points = detector.detect_and_delete(pcd_path, position['x'], position['y'])
                    
                    clear_pcd = pcd_from_points(clear_points)
                    clear_pcd.save(f"{pcd_deleted_full_path}/{pcd_file}")
                elif pcd_file.endswith('.json'):
                    file_path = f'{pcd_folder_path}/{pcd_file}'
                    shutil.copy(file_path, pcd_deleted_full_path)
                    



if __name__ == "__main__":
    main()
