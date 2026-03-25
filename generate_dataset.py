import pickle
import pandas as pd
import numpy as np
from pypcd4 import PointCloud
import cv2
from PIL import Image
import configparser
from utils import *
import os
from tqdm import tqdm


config = configparser.ConfigParser()
config.read('settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)

BEV_LABELS = ast.literal_eval(config['CONSTANTS']['BEV_LABELS'])

PANDASET_PATH = config['PATHS']['PANDASET_PATH']
BEV_DATASET_PATH = config['PATHS']['BEV_DATASET_PATH']
YOLO_DATASET_PATH = config['PATHS']['YOLO_DATASET_PATH']

NEED_SHUFFLE = bool(config['BEV_SHUFFLE']['NEED_SHUFFLE'])


def write_imgs(imgs, imgs_path, pandaset_folder_num, file):
    for idx in range(len(imgs)):
        img = cv2.cvtColor(imgs[idx].astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        write_path = f"{imgs_path}/{pandaset_folder_num}_{file[:-4]}_{idx:02d}.jpg"
        cv2.imwrite(write_path, img)
    
def write_labels(all_labels, labels_path, pandaset_folder_num, file):
    def labels_to_text(labels):
        label_text = ''
        for label in labels:
            label_text += f"{int(label[0])} {" ".join(label[1:].astype(str))}\n"
        return label_text

    for idx in range(len(all_labels)):
        labels = all_labels[idx]
        label_text = labels_to_text(labels)
        
        write_path = f"{labels_path}/{pandaset_folder_num}_{file[:-4]}_{idx:02d}.txt"
        with open(write_path, 'w') as labelfile:
            labelfile.write(label_text)

def get_bev_imgs_with_labels(cuboids_array, points_array):
    x_min_border = -WM
    x_max_border = WM

    y_min_border = -HM
    y_max_border = HM
    
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
            
            img_map = pcd_to_img_map(cur_area_point_arr, row, col, x_min_border, y_min_border, z_min, z_max)
            bevImage = img_map * 255
            images.append(bevImage)
            
            all_labels.append(cur_area_label_arr)
    
    return images, all_labels
            
            
def create_BEV_dataset():
    if not os.path.exists(PANDASET_PATH):
        print('Неверный путь к Pandaset')

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
            
            imgs, all_labels = get_bev_imgs_with_labels(cuboids_array, points_array)
            
            write_imgs(imgs, images_path, pandaset_folder_num, file)
            write_labels(all_labels, labels_path, pandaset_folder_num, file)

            
        
        


    

def main():
    if not os.path.exists(BEV_DATASET_PATH):
        create_BEV_dataset()
    
    if NEED_SHUFFLE:
        create_yolo_data()


if __name__ == "__main__":
    main()
