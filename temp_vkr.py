import numpy as np


BEV_LABELS = ['Car', 'Pedestrian', 'Bus']
PCD_A_W = 50
PCD_A_H = 50


def get_label_indexes(label_col):
    label_indexes = np.asarray([BEV_LABELS.index(x) for x in label_col])
    return label_indexes

def xywhr_to_4xy(xywhr):
    def rotate_x(a):
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        return lambda x, y: x * cos_a - y * sin_a

    def rotate_y(a):
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        return lambda x, y: x * sin_a + y * cos_a
    
    w = xywhr[:, 2]
    h = xywhr[:, 3]
    yaw = xywhr[:, 4].reshape(xywhr.shape[0], -1)
    
    hw = w / 2
    hh = h / 2
    
    corner_rel_xs = np.asarray([-hw, hw, hw, -hw]).T
    corner_rel_ys = np.asarray([-hh, -hh, hh, hh]).T

    rotate_x_func = rotate_x(yaw)
    rotate_y_func = rotate_y(yaw)
    
    points_4xy = np.zeros((xywhr.shape[0], 8))
    points_4xy[:, 0::2] = rotate_x_func(corner_rel_xs, corner_rel_ys) + w.reshape(xywhr.shape[0], -1)
    points_4xy[:, 1::2] = rotate_y_func(corner_rel_xs, corner_rel_ys) + w.reshape(xywhr.shape[0], -1)

    return points_4xy

def get_labels(cuboid_data):
    label_data = np.zeros((cuboid_data.shape[0], 9))

    
    label_data[:, 0] = get_label_indexes(cuboid_data[:, 1])

    xywhr = np.zeros((cuboid_data.shape[0], 5))
    xywhr[:, :2] = cuboid_data[:, 3:5]
    xywhr[:, 2:4] = cuboid_data[:, 6:8]
    xywhr[:, 4] = cuboid_data[:, 2]
    # print(xywhr)
    cuboid_points = xywhr_to_4xy(xywhr)

    print(label_data)

def normalize_cuboid_xy(cuboid_data, row, col, x_min_border, y_min_border):
    x_min_area_border = x_min_border + row * PCD_A_W
    y_min_area_border = y_min_border + col * PCD_A_H
    
    cuboid_data[:, 3] = (np.float32(cuboid_data[:, 3]) - x_min_area_border) / PCD_A_W
    cuboid_data[:, 4] = (np.float32(cuboid_data[:, 4]) - y_min_area_border) / PCD_A_H
    
    cuboid_data[:, 6] = np.float32(cuboid_data[:, 6]) / PCD_A_W
    cuboid_data[:, 7] = np.float32(cuboid_data[:, 7]) / PCD_A_H

    return cuboid_data

start_cols = np.asarray([['231', 'Car', 1.57, 4, 3, 2, 1, 1, 1],
                         ['123', 'Bus', 1.57, 14, 13, 12, 11, 11, 11],
                         ['aaa', 'Pedestrian', 1.57, 41, 31, 21, 11, 11, 11]])

# norm = normalize_cuboid_xy(start_cols, 0, 0, 0, 0)
# print(norm)

get_labels(start_cols)