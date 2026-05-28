import configparser
import ast
import numpy as np
from pypcd4 import PointCloud


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
config.read('scripts/settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)


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
    
    points_array[:, 0] = np.clip(points_array[:, 0], 0, BW - 1)
    points_array[:, 1] = np.clip(points_array[:, 1], 0, BH - 1)
    
    x_coords = points_array[:, 0].astype(np.int32)
    y_coords = points_array[:, 1].astype(np.int32)

    flat_coords = x_coords * BW + y_coords
    
    unique_coords, indices, counts = np.unique(flat_coords, return_index=True, return_counts=True)
    
    # 2D Координаты
    ux = unique_coords // BW
    uy = unique_coords % BW
    
    intensity_map[ux, uy] = points_array[indices, 3]
    height_map[ux, uy] = points_array[indices, 2]
    density_map[ux, uy] = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    
    img_map = np.stack([intensity_map, height_map, density_map], axis=-1)
    
    return img_map


# Распределение всех точек по необходимым зонам (row_num x col_num)
def get_point_areas(points_array, row_num, col_num, x_min_border, y_min_border):
    point_areas = [[[] for _ in range((col_num))] for _ in range(row_num)]

    x_idx = ((points_array[:, 0] - x_min_border) // PCD_A_W).astype(np.int32)
    y_idx = ((points_array[:, 1] - y_min_border) // PCD_A_H).astype(np.int32)
    
    valid_mask = (x_idx >= 0) & (y_idx >= 0) & (x_idx < row_num) & (y_idx < col_num)
    
    valid_points = points_array[valid_mask]
    valid_x = x_idx[valid_mask]
    valid_y = y_idx[valid_mask]
    
    for r in range(row_num):
        for c in range(col_num):
            mask = (valid_x == r) & (valid_y == c)
            point_areas[r][c] = valid_points[mask]
    
    return point_areas

def get_work_pcd_area(pcd_points):
    mask_x = (pcd_points[:, 0] >= -WM) & (pcd_points[:, 0] <= WM)
    
    mask_y = (pcd_points[:, 1] >= -HM) & (pcd_points[:, 1] <= HM)
    
    valid_points_mask = mask_x & mask_y
    
    return pcd_points[valid_points_mask]

def pcd_from_points(point_array):
    return PointCloud.from_points(point_array,
                                  ['x', 'y', 'z', 'i'],
                                  [np.float32, np.float32, np.float32, np.float32])