import configparser
import numpy as np
import ast
import os
import shutil
import yaml
import cv2
from .general import parse_configs


def check_shuffle_perc():
    if float(config['BEV_SHUFFLE']['TRAIN_PERC']) + float(config['BEV_SHUFFLE']['TEST_PERC']) >= 100:
        raise RuntimeError("Процент Train и Test слишком велик (сумма больше или равна 100 процентам)")
    
    
config = configparser.ConfigParser()
config.read('scripts/settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)

BEV_LABELS = ast.literal_eval(config['CONSTANTS']['BEV_LABELS'])

BEV_DATASET_PATH = config['PATHS']['BEV_DATASET_PATH']
YOLO_DATASET_PATH = config['PATHS']['YOLO_DATASET_PATH']

RANDOM_STATE = int(config['BEV_SHUFFLE']['RANDOM_STATE']) if bool(config['BEV_SHUFFLE']['USE_RANDOM_STATE']) else None

check_shuffle_perc()
TRAIN_PART = float(config['BEV_SHUFFLE']['TRAIN_PERC']) / 100
TEST_PART = float(config['BEV_SHUFFLE']['TEST_PERC']) / 100


def get_label_areas(cuboids_array, row_num, col_num, x_min_border, y_min_border):
    label_areas = [[[] for _ in range((col_num))] for _ in range(row_num)]

    for cuboid in cuboids_array:
        x_idx = int((cuboid[2] - x_min_border) // PCD_A_W)
        y_idx = int((cuboid[3] - y_min_border) // PCD_A_H)
        
        if x_idx >= 0 and y_idx >= 0 and x_idx < row_num and y_idx < col_num:
            normalize_cuboid = get_normalized_cuboid(cuboid, x_idx, y_idx, x_min_border, y_min_border)
            cuboid_labels = get_labels(normalize_cuboid)
            label_areas[x_idx][y_idx].append(cuboid_labels)
    
    return label_areas

def xywhr_to_4xy(xywhr):
    cy, cx, w, h, r = xywhr
    yaw = r - np.pi / 2
    
    cos_a = np.cos(yaw)
    sin_a = np.sin(yaw)
    
    R = np.array([[cos_a, -sin_a], 
                  [sin_a,  cos_a]])
    
    dx, dy = w / 2, h / 2
    corners = np.array([
        [-dx,  dy],
        [ dx,  dy],
        [-dx, -dy],
        [ dx, -dy]
    ])
    
    points = np.round(corners @ R + [cx, cy], 4)
    
    return points.flatten()

def get_label_index(label):
    label_index = BEV_LABELS.index(label)
    return label_index

def get_normalized_cuboid(cuboid_data, row, col, x_min_border, y_min_border):
    x_min_area_border = x_min_border + row * PCD_A_W
    y_min_area_border = y_min_border + col * PCD_A_H
    
    normalized_cuboid = cuboid_data.copy()
    
    normalized_cuboid[2] = (np.float32(normalized_cuboid[2]) - x_min_area_border) / PCD_A_W
    normalized_cuboid[3] = (np.float32(normalized_cuboid[3]) - y_min_area_border) / PCD_A_H
    
    normalized_cuboid[4] = np.float32(normalized_cuboid[4]) / PCD_A_W
    normalized_cuboid[5] = np.float32(normalized_cuboid[5]) / PCD_A_H
    
    return normalized_cuboid

def get_labels(cuboid_data):
    label_data = np.zeros(9)

    label_data[0] = get_label_index(cuboid_data[0])

    xywhr = np.zeros(5)
    xywhr[:4] = cuboid_data[2:6]
    xywhr[4] = cuboid_data[1] 

    cuboid_points = xywhr_to_4xy(xywhr)
    label_data[1:] = cuboid_points

    return label_data

def create_yolo_data_folder(img_files, label_files, folder_name):
    copy_folder = f"{YOLO_DATASET_PATH}/{folder_name}"
    os.mkdir(copy_folder)
    os.mkdir(f"{copy_folder}/images/")
    os.mkdir(f"{copy_folder}/labels/")
    for i in range(len(img_files)):
        shutil.copy(img_files[i], f"{copy_folder}/images/{img_files[i][-13:-10]}_{img_files[i][-9:]}")
        shutil.copy(label_files[i], f"{copy_folder}/labels/{label_files[i][-13:-10]}_{label_files[i][-9:]}")

def create_yaml_file():
    yaml_path = f'{YOLO_DATASET_PATH}/data.yaml'
    yaml_data = {
        'path': os.path.abspath(YOLO_DATASET_PATH),
        'train': 'train/images',
        'val': 'validation/images',
        'test': 'test/images',
        'train_label_dir': "train/labels",
        'val_label_dir': 'validation/labels',
        'test_label_dir': 'test/labels',
        'names': {i: name for i, name in enumerate(BEV_LABELS)}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

def get_img_label_files(folders):
    images_path = f"{BEV_DATASET_PATH}/images"
    labels_path = f"{BEV_DATASET_PATH}/labels"
    
    train_img_files = [f"{images_path}/{x}/{y}"
                       for x in folders
                       for y in os.listdir(f"{images_path}/{x}")]
    train_label_files = [f"{labels_path}/{x}/{y}"
                       for x in folders
                       for y in os.listdir(f"{labels_path}/{x}")]
    
    return (train_img_files, train_label_files)

def get_train_test_val_files(data_files):
    rng = np.random.RandomState(RANDOM_STATE)
    indexes = rng.permutation(np.arange(data_files.size))
    
    train_indexes = indexes[0:int(TRAIN_PART * len(indexes))]
    test_indexes = indexes[len(train_indexes):len(train_indexes) + int(TEST_PART * len(indexes))]
    validation_indexes = indexes[len(train_indexes) + len(test_indexes):]
    
    train_img_lbl = get_img_label_files(data_files[train_indexes])
    test_img_lbl = get_img_label_files(data_files[test_indexes])
    val_img_lbl = get_img_label_files(data_files[validation_indexes])

    return (train_img_lbl, test_img_lbl, val_img_lbl)
    

def create_yolo_data():
    data_files = np.asarray(os.listdir(f"{BEV_DATASET_PATH}/images"))
    data_files.sort()
    
    train_files, test_files, val_files = get_train_test_val_files(data_files)

    if os.path.exists(YOLO_DATASET_PATH):
        shutil.rmtree(YOLO_DATASET_PATH)

    os.makedirs(YOLO_DATASET_PATH, exist_ok=True)
    
    
    create_yolo_data_folder(train_files[0], train_files[1], 'train')
    create_yolo_data_folder(test_files[0], test_files[1], 'test')
    create_yolo_data_folder(val_files[0], val_files[1], 'validation')
    
    create_yaml_file()

def filter_empty_obbs(image, obbs, min_useful_ratio=0.03, black_thresh=10):
    valid_indices = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    for idx, obb in enumerate(obbs):
        pts = obb[1:].reshape(4, 2)
        
        pts_abs = pts * np.array([BW, BH])
        
        pts_abs = np.round(pts_abs).astype(np.int32)
        
        x, y, w, h = cv2.boundingRect(pts_abs)
        
        x_min, y_min = max(0, x), max(0, y)
        x_max, y_max = min(BW, x + w), min(BH, y + h)
        
        if x_max <= x_min or y_max <= y_min:
            continue
        
        roi_gray = gray[y_min:y_max, x_min:x_max]
        
        local_pts = pts_abs - [x_min, y_min]
        
        mask = np.zeros(roi_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [local_pts], 255)
        
        polygon_pixels = roi_gray[mask == 255]
        
        if len(polygon_pixels) == 0:
            continue
            
        useful_pixels_count = np.sum(polygon_pixels > black_thresh)
        useful_ratio = useful_pixels_count / len(polygon_pixels)
        
        if useful_ratio >= min_useful_ratio:
            valid_indices.append(idx)
            
    return obbs[valid_indices]