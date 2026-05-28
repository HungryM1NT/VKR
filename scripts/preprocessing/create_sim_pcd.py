import os
import ast
import shutil
import pickle
import configparser
import json
import numpy as np
from tqdm import tqdm
from ..utils.general import get_work_pcd_area, pcd_from_points


config = configparser.ConfigParser()
config.read('scripts/settings.conf')

PCD_COL = ast.literal_eval(config['CONSTANTS']['PCD_COLUMNS'])
CUB_COL = ['yaw', 'position.x', 'position.y', 'position.z', 'dimensions.x', 'dimensions.y', 'dimensions.z']

PCD_PATH = config['PATHS']['PCD_PATH']
PCD_PATH += "_simulated"
PANDASET_PATH = config['PATHS']['PANDASET_PATH']

def xyzwlha_to_4xy2z(cuboids_array):
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

def move_and_rotate_cuboids(pcd_points, cuboids_array, coords, z_mins, z_maxs, 
                            move_radius=5.0, max_angle=np.pi/4):

    xy_points = pcd_points[:, :2]
    z_points = pcd_points[:, 2]
    
    global_dynamic_mask = np.zeros(len(pcd_points), dtype=bool)
    moved_points_list = []
    
    for i, poly in enumerate(coords):
        A, B, C = poly[0], poly[1], poly[2]
        
        AB = B - A
        BC = C - B
        
        AP = xy_points - A
        BP = xy_points - B
        
        dot_AP_AB = np.dot(AP, AB)
        dot_AB_AB = np.dot(AB, AB)
        
        dot_BP_BC = np.dot(BP, BC)
        dot_BC_BC = np.dot(BC, BC)
        
        inside_mask = (0 <= dot_AP_AB) & (dot_AP_AB <= dot_AB_AB) & \
                      (0 <= dot_BP_BC) & (dot_BP_BC <= dot_BC_BC)
        
        z_min = z_mins[i]
        z_max = z_maxs[i]
        
        inside_mask &= (z_points >= z_min) & (z_points <= z_max)
        
        global_dynamic_mask |= inside_mask
        
        cuboid_pts = pcd_points[inside_mask].copy()
        
        if len(cuboid_pts) > 0:
            cx = cuboids_array[i, 1]
            cy = cuboids_array[i, 2]
            
            delta_x = np.random.uniform(-move_radius, move_radius)
            delta_y = np.random.uniform(-move_radius, move_radius)
            delta_yaw = np.random.uniform(-max_angle, max_angle)
            
            translated_x = cuboid_pts[:, 0] - cx
            translated_y = cuboid_pts[:, 1] - cy
            
            cos_y = np.cos(delta_yaw)
            sin_y = np.sin(delta_yaw)
            
            rotated_x = translated_x * cos_y - translated_y * sin_y
            rotated_y = translated_x * sin_y + translated_y * cos_y
            
            cuboid_pts[:, 0] = rotated_x + cx + delta_x
            cuboid_pts[:, 1] = rotated_y + cy + delta_y
            
            moved_points_list.append(cuboid_pts)

    static_points = pcd_points[~global_dynamic_mask]
    
    if moved_points_list:
        final_points = np.vstack([static_points] + moved_points_list)
    else:
        final_points = static_points
        
    return final_points


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
                    cuboids_array = cuboids_data.to_numpy()
                
                with open(f'{lidar_path}/{file}', 'rb') as f:
                    lidar_data = pickle.load(f)
                    points_array = lidar_data[PCD_COL].to_numpy(dtype=np.float32)
                    points_array = get_work_pcd_area(points_array)
                
                dyn_objects_dict = xyzwlha_to_4xy2z(cuboids_array)
            
                points_array = move_and_rotate_cuboids(
                    points_array,
                    cuboids_array,
                    **dyn_objects_dict,
                    move_radius=6.0,
                    max_angle=np.pi/4
                )

                pcd = pcd_from_points(points_array)
                
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