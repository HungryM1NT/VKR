import os
import ast
import shutil
import pickle
import configparser
from tqdm import tqdm
from pypcd4 import PointCloud
import numpy as np


config = configparser.ConfigParser()
config.read('settings.conf')

PCD_COL = ast.literal_eval(config['CONSTANTS']['PCD_COLUMNS'])

PANDASET_PATH = config['PATHS']['PANDASET_PATH']

TYPES = (np.float32, np.float32, np.float32, np.float32)


def create_HD_map():
    hd_map_points = [np.ndarray(shape=(0, 4), dtype=np.float32)]
    
    if not os.path.exists(PANDASET_PATH):
        print('Неверный путь к PandaSet')
    
    # pandaset_folders = os.listdir(PANDASET_PATH)
    pandaset_folders = ['001']
    for pandaset_folder_num in tqdm(pandaset_folders, desc="Pandaset folders", position=0):
        pandaset_folder_path = f'{PANDASET_PATH}/{pandaset_folder_num}'
        lidar_path = f'{pandaset_folder_path}/lidar'
        
        if not os.path.exists(lidar_path):
            print(f"[W]: У {pandaset_folder_num} отсутствует lidar")
            continue
        
        files = os.listdir(lidar_path)
        
        for file in files:
            if file.endswith('.pkl'):
                with open(f'{lidar_path}/{file}', 'rb') as f:
                    lidar_data = pickle.load(f)
                    points_array = lidar_data[PCD_COL].to_numpy(dtype=np.float32)
                    
                    hd_map_points = np.concatenate((hd_map_points, points_array), axis=0)
                
    hd_map = PointCloud.from_points(hd_map_points, PCD_COL, TYPES)
    write_path = f"HD_map.pcd"
    hd_map.save(write_path)

def main():
    create_HD_map()


if __name__ == "__main__":
    main()
