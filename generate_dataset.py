import pickle
import pandas as pd
import numpy as np
from pypcd4 import PointCloud
import cv2
from PIL import Image
import configparser
from utils import *
import os


config = configparser.ConfigParser()
config.read('settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)
BEV_LABELS = ast.literal_eval(config['CONSTANTS']['BEV_LABELS'])

PANDASET_PATH = './training/Pandaset'
BEV_DATASET_PATH = './training/BEV_Dataset'

def write_imgs(imgs, imgs_path, pandaset_folder_num, file):
    for idx in range(len(imgs)):
        img = cv2.cvtColor(imgs[idx].astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        write_path = f"{imgs_path}/{pandaset_folder_num}_{file[:-4]}_{idx:02d}.jpg"
        cv2.imwrite(write_path, img)
    # img = cv2.cvtColor(bevImage.astype('float32'), cv.COLOR_RGB2BGR)
    #     cv.imwrite(imgSavePath, img)

def get_bev_imgs_with_labels(cuboids_array, points_array):
    z_min = np.min(points_array[:, 2])
    z_max = np.max(points_array[:, 2])
    
    x_min = np.min(cuboids_array[:, 2])
    x_max = np.max(cuboids_array[:, 2])
    x_max_dim = np.max(cuboids_array[:, 4])

    y_min = np.min(cuboids_array[:, 3])
    y_max = np.max(cuboids_array[:, 3])
    y_max_dim = np.max(cuboids_array[:, 5])

    x_min_border = max(x_min - x_max_dim, -WM)
    x_max_border = min(x_max + x_max_dim, WM)

    y_min_border = max(y_min - y_max_dim, -HM)
    y_max_border = min(y_max + y_max_dim, HM)
    
    row_num = int((x_max_border - x_min_border - 1) // PCD_A_W + 1)
    col_num = int((y_max_border - y_min_border - 1) // PCD_A_H + 1)
    
    point_areas = get_point_areas(points_array, row_num, col_num, x_min_border, y_min_border)
    cuboid_areas = get_cuboid_areas(cuboids_array, row_num, col_num, x_min_border, y_min_border)
    
    images = []
    labels = []
    
    for row in range(row_num):
        for col in range(col_num):
            cur_area_point_arr = np.asarray(point_areas[row][col])
            cur_area_cuboid_arr = np.asarray(cuboid_areas[row][col])
            
            if len(cur_area_cuboid_arr) * len(cur_area_point_arr) == 0:
                continue
            
            img_map = pcd_to_img_map(cur_area_point_arr, row, col, x_min_border, y_min_border, z_min, z_max)
            bevImage = img_map * 255
            images.append(bevImage)
            
            area_labels = get_labels(cur_area_cuboid_arr)
            labels.append(area_labels)
            
            # labels_in_one_area = np.zeros((len(cur_area_cuboid_arr), 9))
            # for idx in range(len(cur_area_cuboid_arr)):
            #     cuboid = cur_area_cuboid_arr[idx]
            #     points = xywhr_to_xy4(cuboid[3], cuboid[4], cuboid[6], cuboid[7], cuboid[2])
            #     labels_in_one_area
            #     labels.append([cuboid[1]] + points)
    
    return images, labels
            
            
def create_BEV_dataset():
    if not os.path.exists(PANDASET_PATH):
        print('Неверный путь к Pandaset')

    os.mkdir(BEV_DATASET_PATH)
    images_path = f"{BEV_DATASET_PATH}/images"
    labels_path = f"{BEV_DATASET_PATH}/labels"
    os.mkdir(images_path)
    os.mkdir(labels_path)
    
    pandaset_folders = os.listdir(PANDASET_PATH)
    pandaset_folders.remove('047') # Идея с локализацией
    pandaset_folders = ['001']
    for pandaset_folder_num in pandaset_folders:
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
            with open(f'{cuboids_path}\{file}', 'rb') as f:
                cuboids_data = pickle.load(f)
                cuboids_data = cuboids_data[CUB_COL]
                cuboids_data = cuboids_data[cuboids_data['label'].isin(BEV_LABELS)]
                cuboids_array = cuboids_data.to_numpy()

            with open(f'{lidar_path}\{file}', 'rb') as f:
                lidar_data = pickle.load(f)
                points_array = lidar_data[PCD_COL].to_numpy()
            
            imgs, labels = get_bev_imgs_with_labels(cuboids_array, points_array)
            
            write_imgs(imgs, images_path, pandaset_folder_num, file)
            print(labels)
            
        
        


    

def main():
    if not os.path.exists(BEV_DATASET_PATH):
        create_BEV_dataset()



if __name__ == "__main__":
    main()

    # 32 7 7 1