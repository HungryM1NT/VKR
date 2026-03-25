import configparser
import numpy as np
import ast


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
    
    
config = configparser.ConfigParser()
config.read('settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)

BEV_LABELS = ast.literal_eval(config['CONSTANTS']['BEV_LABELS'])

    
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


def pcd_to_img_map(points_array, x_idx, y_idx, x_min_border, y_min_border, z_min, z_max):
    x_min_area_border = x_min_border + x_idx * PCD_A_W
    y_min_area_border = y_min_border + y_idx * PCD_A_H

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
