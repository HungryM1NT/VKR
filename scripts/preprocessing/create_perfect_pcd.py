import os
import ast
import shutil
import pickle
import configparser
import json
import numpy as np
from tqdm import tqdm
from pypcd4 import PointCloud
from ..utils.general import get_work_pcd_area


config = configparser.ConfigParser()
config.read('settings.conf')

PCD_COL = ast.literal_eval(config['CONSTANTS']['PCD_COLUMNS'])
CUB_COL = ['yaw', 'position.x', 'position.y', 'position.z', 'dimensions.x', 'dimensions.y', 'dimensions.z']

PCD_PATH = config['PATHS']['PCD_PATH']
PCD_PATH += "_perfect"
PANDASET_PATH = config['PATHS']['PANDASET_PATH']

TYPES = (np.float32, np.float32, np.float32, np.float32)

def xyzwlha_to_4xy2z(cuboids_array):
    """_summary_

    Args:
        cuboids_array: [[yaw, x, y, z, w, l, h]]
    Return:
        {
            coords: [[x0, y0, x1, y1, x2, y2, x3, y3]]
            z_mins: [z_min]
            z_maxs: [z_max]
        }
    """
    
    coords = []
    z_mins = []
    z_maxs = []
    
    for cuboid in cuboids_array:
        yaw = cuboid[0]
        cx = cuboid[1]
        cy = cuboid[2]
        cz = cuboid[3]
        l = cuboid[4]
        w = cuboid[5]
        h = cuboid[6]
        
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        
        dx = l / 2.0
        dy = w / 2.0
        
        x1, y1 = dx, dy
        x2, y2 = dx, -dy
        x3, y3 = -dx, -dy
        x4, y4 = -dx, dy
        
        def rotate_and_translate(lx, ly):
            gx = lx * cos_y - ly * sin_y + cx
            gy = lx * sin_y + ly * cos_y + cy
            return [gx, gy]
            
        p1 = rotate_and_translate(x1, y1)
        p2 = rotate_and_translate(x2, y2)
        p3 = rotate_and_translate(x3, y3)
        p4 = rotate_and_translate(x4, y4)
        
        cuboid_points = np.array([p1, p2, p3, p4])
        
        z_min = cz - h / 2.0 - 0.5
        z_max = cz + h / 2.0 + 1.0
        
        coords.append(cuboid_points)
        z_mins.append(z_min)
        z_maxs.append(z_max)
    
    return {
        "coords": np.asarray(coords),
        "z_mins": z_mins,
        "z_maxs": z_maxs
    }

def get_3d_delete_mask(pcd_points, coords, z_mins, z_maxs, ego_center=None, ego_size=(4.0, 4.0)):
    xy_points = pcd_points[:, :2]
    
    z_points = pcd_points[:, 2]
    
    mask = np.zeros(len(pcd_points), dtype=bool)
    
    for i, poly in enumerate(coords):
        # Берем первые три точки прямоугольника (они идут последовательно)
        A, B, C = poly[0], poly[1], poly[2]
        
        # Векторы сторон
        AB = B - A
        BC = C - B
        
        # Векторы от углов до всех точек облака
        AP = xy_points - A
        BP = xy_points - B
        
        # Скалярные произведения (векторные проекции)
        dot_AP_AB = np.dot(AP, AB)
        dot_AB_AB = np.dot(AB, AB)
        
        dot_BP_BC = np.dot(BP, BC)
        dot_BC_BC = np.dot(BC, BC)
        
        # Точка внутри 2D-прямоугольника (сверху, BEV-проекция)
        inside_mask = (0 <= dot_AP_AB) & (dot_AP_AB <= dot_AB_AB) & \
                      (0 <= dot_BP_BC) & (dot_BP_BC <= dot_BC_BC)
        

        z_min = z_mins[i]
        z_max = z_maxs[i]
        
        inside_mask &= (z_points >= z_min) & (z_points <= z_max)
                    
        mask |= inside_mask
        
    if ego_center is not None:
        x_center, y_center = ego_center
        size_x, size_y = ego_size
        
        x_min = x_center - (size_x / 2.0)
        x_max = x_center + (size_x / 2.0)
        y_min = y_center - (size_y / 2.0)
        y_max = y_center + (size_y / 2.0)
        
        ego_mask = (pcd_points[:, 0] >= x_min) & (pcd_points[:, 0] <= x_max) & \
                   (pcd_points[:, 1] >= y_min) & (pcd_points[:, 1] <= y_max)
                   
        mask |= ego_mask
        
    return np.invert(mask)

def transform_pkl_to_pcd():
    if not os.path.exists(PANDASET_PATH):
        print('Неверный путь к PandaSet')

    os.makedirs(PCD_PATH, exist_ok=True)
    
    pandaset_folders = os.listdir(PANDASET_PATH)

    for pandaset_folder_num in tqdm(pandaset_folders, desc="Pandaset folders", position=0):
        pcd_full_path = f"{PCD_PATH}/{pandaset_folder_num}"
        os.makedirs(pcd_full_path, exist_ok=True)
        
        pandaset_folder_path = f'{PANDASET_PATH}/{pandaset_folder_num}'
        lidar_path = f'{pandaset_folder_path}/lidar'
        cuboids_path = f'{pandaset_folder_path}/annotations/cuboids'
        
        if not os.path.exists(lidar_path):
            print(f"[W]: У {pandaset_folder_num} отсутствует lidar")
            continue
        if not os.path.exists(cuboids_path):
            print(f"[W]: У {pandaset_folder_num} отсутствует cuboids")
            continue
        
        files = os.listdir(lidar_path)
        files.sort()
                
        with open(f'{lidar_path}/poses.json') as f:
            json_poses = json.load(f)
        
        for file in files:
            if file.endswith('.pkl'):
                with open(f'{cuboids_path}/{file}', 'rb') as f:
                    cuboids_data = pickle.load(f)
                    cuboids_data = cuboids_data[cuboids_data['cuboids.sensor_id'].isin([-1, 0])]
                    cuboids_data = cuboids_data[CUB_COL]
                    # cuboids_data = cuboids_data[cuboids_data['label'].isin(BEV_LABELS)]
                    cuboids_array = cuboids_data.to_numpy()
                
                with open(f'{lidar_path}/{file}', 'rb') as f:
                    lidar_data = pickle.load(f)
                    points_array = lidar_data[PCD_COL].to_numpy(dtype=np.float32)
                    points_array = get_work_pcd_area(points_array)
                
                position = json_poses[int(file[:-4])]['position']
                
                dyn_objects_dict = xyzwlha_to_4xy2z(cuboids_array)
            
                delete_mask = get_3d_delete_mask(points_array,
                                          **dyn_objects_dict,
                                          ego_center=(position['x'], position['y']), 
                                          ego_size=(4.0, 4.0))
                
                points_array = points_array[delete_mask]

                pcd = PointCloud.from_points(points_array, PCD_COL, TYPES)
                
                write_path = f"{pcd_full_path}/{file[:-4]}.pcd"
                pcd.save(write_path)
            elif file.endswith('.json'):
                file_path = f'{lidar_path}/{file}'
                shutil.copy(file_path, pcd_full_path)

def main():
    if not os.path.exists(PCD_PATH):
        transform_pkl_to_pcd()


if __name__ == "__main__":
    main()
