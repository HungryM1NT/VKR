import numpy as np

def xywhr_to_4xy(xywhr):
    def rotate_x(a):
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        return lambda x, y: x * cos_a - y * sin_a

    def rotate_y(a):
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        return lambda x, y: x * sin_a + y * cos_a

    start_x1 = - xywhr[:, 2] / 2
    start_x2 = xywhr[:, 2] / 2
    start_y1 = xywhr[:, 3] / 2
    start_y2 = -xywhr[:, 3] / 2
    
    # print(start_x1)
    rotate_x_func = rotate_x(xywhr[:, 4])
    rotate_y_func = rotate_y(xywhr[:, 4])
    
    points_4xy = np.zeros((xywhr.shape[0], 8))
    
    points_4xy[:, 0] = np.round(rotate_x_func(start_x1, start_y1), 4) + xywhr[:, 0]
    points_4xy[:, 2] = np.round(rotate_x_func(start_x2, start_y1), 4) + xywhr[:, 0]
    points_4xy[:, 4] = np.round(rotate_x_func(start_x1, start_y2), 4) + xywhr[:, 0]
    points_4xy[:, 6] = np.round(rotate_x_func(start_x2, start_y2), 4) + xywhr[:, 0]
    
    points_4xy[:, 1] = np.round(rotate_y_func(start_x1, start_y1), 4) + xywhr[:, 1]
    points_4xy[:, 3] = np.round(rotate_y_func(start_x2, start_y1), 4) + xywhr[:, 1]
    points_4xy[:, 5] = np.round(rotate_y_func(start_x1, start_y2), 4) + xywhr[:, 1]
    points_4xy[:, 7] = np.round(rotate_y_func(start_x2, start_y2), 4) + xywhr[:, 1]

    return points_4xy

def xywhr_to_4xy2(xywhr):
    def rotate_x(a):
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        return lambda x, y: x * cos_a - y * sin_a

    def rotate_y(a):
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        return lambda x, y: x * sin_a + y * cos_a
    
    x = xywhr[:, 2].reshape(xywhr.shape[0], -1)
    y = xywhr[:, 3].reshape(xywhr.shape[0], -1)
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
    points_4xy[:, 0::2] = rotate_x_func(corner_rel_xs, corner_rel_ys) + x
    points_4xy[:, 1::2] = rotate_y_func(corner_rel_xs, corner_rel_ys) + y

    return points_4xy

def convert_xywhr_to_8pts(x, y, w, h, yaw):
    """
    Конвертирует ориентированный бокс в 4 угловые точки.
    yaw: угол в радианах.
    """
    # 1. Задаем углы прямоугольника относительно центра (0,0) без поворота
    hw, hh = w / 2, h / 2
    # Порядок: TL, TR, BR, BL (по часовой стрелке)
    points = np.array([
        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
    ])

    # 2. Создаем матрицу поворота
    cos_a = np.cos(yaw)
    sin_a = np.sin(yaw)
    rot_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    # 3. Поворачиваем точки и смещаем к центру (x, y)
    rotated_points = np.dot(points, rot_matrix.T) + [x, y]
    
    # Возвращаем плоский список из 8 координат
    return rotated_points.flatten()

# Пример использования:
x_c, y_c, width, height, angle_rad = 0.5, 0.5, 0.2, 0.1, 0.785  # 45 градусов
pts8 = convert_xywhr_to_8pts(x_c, y_c, width, height, angle_rad)

print("Формат для YOLO OBB (8 точек):")
print(" ".join(map(lambda x: f"{x:.6f}", pts8)))

temp = xywhr_to_4xy(np.asarray([[0.5, 0.5, 0.2, 0.1, 0.785]]))
print(temp)

temp = xywhr_to_4xy2(np.asarray([[0.5, 0.5, 0.2, 0.1, 0.785]]))
print(temp)