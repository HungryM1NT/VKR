import configparser
import numpy as np
import ast
import os
import shutil
import yaml
import cv2


def parse_configs(config):
    constants = config['CONSTANTS']
    cuboid_columns = ast.literal_eval(constants['CUBOIDS_COLUMNS'])
    pcd_columns = ast.literal_eval(constants['PCD_COLUMNS'])

    pcd_area_width = int(constants['PCD_AREA_WIDTH'])
    pcd_area_height = int(constants['PCD_AREA_HEIGHT'])

    bev_width = int(constants['BEV_WIDTH'])
    bev_height = int(constants['BEV_HEIGHT'])

    width_magic = int(constants['WIDTH_MAGIC'])
    height_magic = int(constants['HEIGHT_MAGIC'])
    return (cuboid_columns, pcd_columns,
            pcd_area_width, pcd_area_height,
            bev_width, bev_height,
            width_magic, height_magic)

def check_shuffle_perc():
    if float(config['BEV_SHUFFLE']['TRAIN_PERC']) + float(config['BEV_SHUFFLE']['TEST_PERC']) >= 100:
        raise RuntimeError("Процент Train и Test слишком велик (сумма больше или равна 100 процентам)")
    
    
config = configparser.ConfigParser()
config.read('settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)

BEV_LABELS = ast.literal_eval(config['CONSTANTS']['BEV_LABELS'])

BEV_DATASET_PATH = config['PATHS']['BEV_DATASET_PATH']
YOLO_DATASET_PATH = config['PATHS']['YOLO_DATASET_PATH']

RANDOM_STATE = int(config['BEV_SHUFFLE']['RANDOM_STATE']) if bool(config['BEV_SHUFFLE']['USE_RANDOM_STATE']) else None

check_shuffle_perc()
TRAIN_PART = float(config['BEV_SHUFFLE']['TRAIN_PERC']) / 100
TEST_PART = float(config['BEV_SHUFFLE']['TEST_PERC']) / 100

    
# Распределение всех точек по необходимым зонам (row_num x col_num)
def get_point_areas(points_array, row_num, col_num, x_min_border, y_min_border):
    point_areas = [[[] for _ in range((col_num))] for _ in range(row_num)]

    for point in points_array:
        
        x_idx = int((point[0] - x_min_border) // PCD_A_W)
        y_idx = int((point[1] - y_min_border) // PCD_A_H)
        if x_idx >= 0 and y_idx >= 0 and x_idx < row_num and y_idx < col_num:
            point_areas[x_idx][y_idx].append(point)
    
    return point_areas

# Распределение всех лейблов по необходимым зонам (row_num x col_num)
def get_label_areas(cuboids_array, row_num, col_num, x_min_border, y_min_border):
    label_areas = [[[] for _ in range((col_num))] for _ in range(row_num)]

    for cuboid in cuboids_array:
        x_idx = int((cuboid[2] - x_min_border) // PCD_A_W)
        y_idx = int((cuboid[3] - y_min_border) // PCD_A_H)
        if x_idx >= 0 and y_idx >= 0 and x_idx < row_num and y_idx < col_num:
            normalize_cuboid = get_normalized_cuboid(cuboid, x_idx, y_idx, x_min_border, y_min_border)
            cuboid_labels = get_labels(normalize_cuboid)
            if np.min(cuboid_labels[1:]) < 0 or np.max(cuboid_labels[1:]) > 1:
                continue
            label_areas[x_idx][y_idx].append(cuboid_labels)
    
    return label_areas

# Create (x, y): (counts, maxZ) dict
def get_CoordToCountValInt_dict(points):
    coord_to_countval = dict()
    for point in points:
        if coord_to_countval.get((point[0], point[1])) == None:
            coord_to_countval[((point[0], point[1]))] = [1, point[2], point[3]]
        else:
            coord_to_countval[((point[0], point[1]))][0] += 1
    return coord_to_countval


def pcd_to_img_map(points_array, x_idx, y_idx, x_min_border, y_min_border, z_min, z_max, x_stride, y_stride):
    x_min_area_border = x_min_border + x_idx * x_stride
    y_min_area_border = y_min_border + y_idx * y_stride

    # Нормализуем x,y [0: 1]
    points_array[:, 0] = (points_array[:, 0] - x_min_area_border) / PCD_A_W
    points_array[:, 1] = (points_array[:, 1] - y_min_area_border) / PCD_A_H

    # Приводим x,y к [0: BEV_H/BEV_W]
    points_array[:, 0] = np.int32(points_array[:, 0] * BW)
    points_array[:, 1] = np.int32(points_array[:, 1] * BH)

    # Приводим z к [0, 1]
    points_array[:, 2] = (points_array[:, 2] - z_min) / (z_max - z_min)

    # Сортируем (увел х, увел y, умен z)
    ix = np.lexsort((-points_array[:, 2], points_array[:, 1], points_array[:, 0]))
    points_array = points_array[ix]

    height_map = np.zeros((BH, BW))
    density_map = np.zeros((BH, BW))
    intensity_map = np.zeros((BH, BW))

    points_array[:, 0] = np.minimum(np.maximum(points_array[:, 0], 0), BH - 1)
    points_array[:, 1] = np.minimum(np.maximum(points_array[:, 1], 0), BH - 1)

    coord_to_countval = get_CoordToCountValInt_dict(points_array)

    for ((x, y), (c, z, i)) in coord_to_countval.items():
        density_map[int(x)][int(y)] = min(1.0, np.log(c + 1) / np.log(64))
        height_map[int(x)][int(y)] = z
        intensity_map[int(x)][int(y)] = i

    img_map = np.zeros([BH, BW, 3])
    img_map[:,:,0] = density_map
    img_map[:,:,1] = height_map
    img_map[:,:,2] = intensity_map
    
    return img_map

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