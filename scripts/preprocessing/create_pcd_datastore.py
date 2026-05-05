import os
import ast
import shutil
import pickle
import configparser
from tqdm import tqdm
import numpy as np
from utils.general import pcd_from_points


config = configparser.ConfigParser()
config.read('settings.conf')

PCD_COL = ast.literal_eval(config['CONSTANTS']['PCD_COLUMNS'])

PCD_PATH = config['PATHS']['PCD_PATH']
PANDASET_PATH = config['PATHS']['PANDASET_PATH']


def transform_pkl_to_pcd():
    if not os.path.exists(PANDASET_PATH):
        print('Неверный путь к PandaSet')

    os.makedirs(PCD_PATH, exist_ok=True)
    
    pandaset_folders = os.listdir(PANDASET_PATH)

    for pandaset_folder_num in tqdm(pandaset_folders, desc="Pandaset folders", position=0):
        pcd_full_path = f"{PCD_PATH}/{pandaset_folder_num}"
        os.makedirs(pcd_full_path, exist_ok=True)
        
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
                
                pcd = pcd_from_points(points_array)
                
                write_path = f"{pcd_full_path}/{file[:-4]}.pcd"
                pcd.save(write_path)
            elif file.endswith('.json'):
                file_path = f'{lidar_path}/{file}'
                shutil.copy(file_path, pcd_full_path)

def main():
    if not os.path.exists(PCD_PATH):
        transform_pkl_to_pcd()


if __name__ == "__main__":
    main()
