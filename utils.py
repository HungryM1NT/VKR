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
def get_cuboid_areas(cuboids_array, row_num, col_num, x_min_border, y_min_border):
    cuboid_areas = [[[] for _ in range((col_num))] for _ in range(row_num)]

    for cuboid in cuboids_array:
        x_idx = int((cuboid[3] - x_min_border) // PCD_A_W)
        y_idx = int((cuboid[4] - y_min_border) // PCD_A_H)
        if x_idx >= 0 and y_idx >= 0 and x_idx < row_num and y_idx < col_num:
            cuboid_areas[x_idx][y_idx].append(cuboid)
    
    return cuboid_areas

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

def xywhr_to_xy4(x, y, w, h, yaw):
    hw, hh = w / 2, h / 2
    points = np.array([
        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
    ])
    
    cos_a = np.cos(yaw)
    sin_a = np.sin(yaw)
    rot_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    rotated_points = np.dot(points, rot_matrix.T) + [x, y]
    
    # return rotated_points.flatten()
    return rotated_points.flatten()


def get_relative_cuboid_arr(cur_area_cuboid_arr, x_idx, y_idx, x_min_border, y_min_border):
    x_min_area_border = x_min_border + x_idx * PCD_A_W
    y_min_area_border = y_min_border + y_idx * PCD_A_H
    
    cur_area_cuboid_arr[:, 3] = (cur_area_cuboid_arr[:, 3] - x_min_area_border) / PCD_A_W
    cur_area_cuboid_arr[:, 4] = (cur_area_cuboid_arr[:, 4] - y_min_area_border) / PCD_A_H
    
    cur_area_cuboid_arr[:, 6] = cur_area_cuboid_arr[:, 6] / PCD_A_W
    cur_area_cuboid_arr[:, 7] = cur_area_cuboid_arr[:, 7] / PCD_A_H

    return cur_area_cuboid_arr



# x_c, y_c, width, height, angle_rad = 3.947, -6.01, 1.841, 5.174, -1.5421142437145487
# x_c, y_c, width, height, angle_rad = 0.5, 0.5, 0.2, 0.1, 0.785  # 45 градусов
# pts8 = xywhr_to_xy4(x_c, y_c, width, height, angle_rad)

# # print("Формат для YOLO OBB (8 точек):")
# print(pts8)