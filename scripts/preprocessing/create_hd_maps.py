import os
import ast
import configparser
from tqdm import tqdm
from .hd_map_creator import create_HD_map


config = configparser.ConfigParser()
config.read('scripts/settings.conf')

PCD_COL = ast.literal_eval(config['CONSTANTS']['PCD_COLUMNS'])
PCD_PATH = config['PATHS']['PCD_PATH']
# PCD_PATH += '_perfect'
# PCD_PATH += '_DELETED'
# PERFECT_HD_MAP_PATH = './training/HD_perfect'
# PERFECT_HD_MAP_PATH = './training/HD_my'
PERFECT_HD_MAP_PATH = './training/HD_bad'
PANDASET_PATH = config['PATHS']['PANDASET_PATH']

def create_clear_hd_maps():
    if not os.path.exists(PCD_PATH):
        print('Нет чистых pcd данных')

    os.makedirs(PERFECT_HD_MAP_PATH, exist_ok=True)
    
    pcd_folders = os.listdir(PCD_PATH)

    for pcd_folder_num in tqdm(pcd_folders, desc="PCD folders", position=0):
        pcds_folder_path = f'{PCD_PATH}/{pcd_folder_num}'
        save_path = f'{PERFECT_HD_MAP_PATH}/{pcd_folder_num}.pcd'

        create_HD_map(pcds_folder_path, save_path)

def main():
    if not os.path.exists(PERFECT_HD_MAP_PATH):
        create_clear_hd_maps()


if __name__ == "__main__":
    main()
