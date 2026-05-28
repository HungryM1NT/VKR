import os
import json
import cv2
import pickle
import configparser
import numpy as np
import ast
from ..utils.general import *
from ..utils.preprocess_utils import *
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

config = configparser.ConfigParser()
config.read('scripts/settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)

BEV_LABELS = ast.literal_eval(config['CONSTANTS']['BEV_LABELS'])

PANDASET_PATH = config['PATHS']['PANDASET_PATH']
BEV_DATASET_PATH = config['PATHS']['BEV_DATASET_PATH']
YOLO_DATASET_PATH = config['PATHS']['YOLO_DATASET_PATH']

NEED_SHUFFLE = bool(config['BEV_SHUFFLE']['NEED_SHUFFLE'])


def write_imgs(imgs, imgs_path, pandaset_folder_num, file):
    images_full_path = f"{imgs_path}/{pandaset_folder_num}"
    os.makedirs(images_full_path, exist_ok=True)
    
    for idx in range(len(imgs)):
        img = imgs[idx]
        
        write_path = f"{images_full_path}/{file[:-4]}_{idx:02d}.png"
        cv2.imwrite(write_path, img)
    
def write_labels(all_labels, labels_path, pandaset_folder_num, file):
    def labels_to_text(labels):
        label_text = ''
        for label in labels:
            label_text += f"{int(label[0])} {" ".join(label[1:].astype(str))}\n"
        return label_text

    labels_full_path = f"{labels_path}/{pandaset_folder_num}"
    os.makedirs(labels_full_path, exist_ok=True)
    
    for idx in range(len(all_labels)):
        labels = all_labels[idx]
        label_text = labels_to_text(labels)
        
        write_path = f"{labels_full_path}/{file[:-4]}_{idx:02d}.txt"
        with open(write_path, 'w') as labelfile:
            labelfile.write(label_text)

def get_bev_imgs_with_labels(cuboids_array, points_array, position):
    x_center = position['x']
    y_center = position['y']
        
    x_min_border = x_center - WM
    x_max_border = x_center + WM

    y_min_border = y_center - HM
    y_max_border = y_center + HM
    
    z_min = np.min(points_array[:, 2])
    z_max = np.max(points_array[:, 2])
    
    row_num = int((x_max_border - x_min_border - 1) // PCD_A_W + 1)
    col_num = int((y_max_border - y_min_border - 1) // PCD_A_H + 1)
    
    point_areas = get_point_areas(points_array, row_num, col_num, x_min_border, y_min_border)
    label_areas = get_label_areas(cuboids_array, row_num, col_num, x_min_border, y_min_border)
    
    images = []
    all_labels = []
    
    for row in range(row_num):
        for col in range(col_num):
            cur_area_point_arr = np.asarray(point_areas[row][col])
            cur_area_label_arr = np.asarray(label_areas[row][col])
            
            if len(cur_area_label_arr) * len(cur_area_point_arr) == 0:
                continue
            
            img_map = pcd_to_img_map(cur_area_point_arr, row, col, x_min_border, y_min_border, z_min, z_max, PCD_A_W, PCD_A_H)
            bevImage = (img_map * 255).astype(np.uint8)
            
            # Фильтр на пустые ббоксы
            area_labels = cur_area_label_arr.copy()
            area_labels = filter_empty_obbs(bevImage, area_labels)
            if len(area_labels) == 0:
                continue
            
            images.append(bevImage)
            all_labels.append(area_labels)
    
    return images, all_labels
            
            
def create_BEV_dataset():
    if not os.path.exists(PANDASET_PATH):
        print('Неверный путь к PandaSet')

    os.makedirs(BEV_DATASET_PATH, exist_ok=True)
    images_path = f"{BEV_DATASET_PATH}/images"
    labels_path = f"{BEV_DATASET_PATH}/labels"
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    
    pandaset_folders = os.listdir(PANDASET_PATH)
    pandaset_folders.remove('047') # Идея с локализацией

    for pandaset_folder_num in tqdm(pandaset_folders, desc="Pandaset folders", position=0):
        pandaset_folder_path = f'{PANDASET_PATH}/{pandaset_folder_num}'
        lidar_path = f'{pandaset_folder_path}/lidar'
        cuboids_path = f'{pandaset_folder_path}/annotations/cuboids'

        if not os.path.exists(lidar_path):
            print(f"[W]: У {pandaset_folder_num} отсутствует lidar")
            continue
        if not os.path.exists(cuboids_path):
            print(f"[W]: У {pandaset_folder_num} отсутствует cuboids")
            continue
        
        files = os.listdir(cuboids_path)
        files.sort()
        
        with open(f'{lidar_path}/poses.json') as f:
            json_poses = json.load(f)
        
        for file in files:
            with open(f'{cuboids_path}/{file}', 'rb') as f:
                cuboids_data = pickle.load(f)
                cuboids_data = cuboids_data[cuboids_data['cuboids.sensor_id'].isin([-1, 0])]
                cuboids_data = cuboids_data[CUB_COL]
                cuboids_data = cuboids_data[cuboids_data['label'].isin(BEV_LABELS)]
                cuboids_array = cuboids_data.to_numpy()

            with open(f'{lidar_path}/{file}', 'rb') as f:
                lidar_data = pickle.load(f)
                points_array = lidar_data[PCD_COL].to_numpy()
                
            file_idx= files.index(file)
            position = json_poses[file_idx]['position']
            
            imgs, all_labels = get_bev_imgs_with_labels(cuboids_array, points_array, position)
            
            write_imgs(imgs, images_path, pandaset_folder_num, file)
            write_labels(all_labels, labels_path, pandaset_folder_num, file)

            

def main():
    if not os.path.exists(BEV_DATASET_PATH):
        create_BEV_dataset()
    
    if NEED_SHUFFLE:
        create_yolo_data()


if __name__ == "__main__":
    main()
