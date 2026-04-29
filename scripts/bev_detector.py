import time
import numpy as np
import configparser
from pypcd4 import PointCloud
from utils.general import parse_configs, get_point_areas, pcd_to_img_map
from utils.inference_utils import get_coords_from_bevs,scale_obbs_percentage, delete_points, get_work_pcd_area


config = configparser.ConfigParser()
config.read('settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)


class BEV_DETECTOR:
    def __init__(self, model):
        self.model = model
        dummy_img = [np.zeros((BH, BW, 3), dtype=np.uint8)]
        model(dummy_img, verbose=False)

    def _read_pcd(self, pcd_path):
        pcd = PointCloud.from_path(pcd_path)
        pcd_points_array = pcd.numpy(("x", "y", "z", "i"))
        points_array = get_work_pcd_area(pcd_points_array)
        return points_array
    
    def get_bev_imgs(self, pcd_points, x_center, y_center):   
        x_min_border = x_center - WM
        x_max_border = x_center + WM

        y_min_border = y_center - HM
        y_max_border = y_center + HM
        
        z_min = np.min(pcd_points[:, 2])
        z_max = np.max(pcd_points[:, 2])
        
        row_num = int((x_max_border - x_min_border - 1) // PCD_A_W + 1)
        col_num = int((y_max_border - y_min_border - 1) // PCD_A_H + 1)
       
        point_areas = get_point_areas(pcd_points, row_num, col_num, x_min_border, y_min_border)

        step_x, step_y = PCD_A_W, PCD_A_H 

        images = []
        borders = []

        for row in range(row_num):
            for col in range(col_num):
                cur_area_point_arr = np.asarray(point_areas[row][col])
                
                if len(cur_area_point_arr) == 0:
                    continue
                
                img_map = pcd_to_img_map(cur_area_point_arr, row, col, x_min_border, y_min_border, z_min, z_max, step_x, step_y)
                bevImage = (img_map * 255).astype(np.uint8)
                
                images.append(bevImage)
                
                cur_x_min_border = x_min_border + row * step_x
                cur_y_min_border = y_min_border + col * step_y
            
                borders.append((cur_x_min_border, cur_y_min_border))
            
        return images, borders
    
    def detect_and_delete(self, pcd_path, x_center=0.0, y_center=0.0):
        self.timer = [time.perf_counter()]

        # Read PCD
        pcd_points = self._read_pcd(pcd_path)        
        self.timer.append(time.perf_counter())
        
        coords = self.detect(pcd_points, x_center, y_center)
        
        filtered_points = self.delete(pcd_points, coords)
        
        return filtered_points
    
    def detect(self, pcd_points, x_center=0.0, y_center=0.0):
        # PCD -> BEV
        bev_images, borders = self.get_bev_imgs(pcd_points, x_center, y_center)
        self.timer.append(time.perf_counter())
        
        # Model predict
        results = []
        for img in bev_images:
            res = self.model(img, verbose=False)
            results.extend(res)
        self.timer.append(time.perf_counter())
    
        all_coords = get_coords_from_bevs(results, borders)
        scaled_coords = scale_obbs_percentage(all_coords, scale_factor=1.2)
        
        # Координаты найденных объектов
        return scaled_coords
       
    def delete(self, pcd_points, object_coords, x_center=0.0, y_center=0.0):
        # Delete points
        filtered_points = delete_points(pcd_points,
                                object_coords,
                                ego_center=(x_center, y_center), 
                                ego_size=(5.0, 5.0))
        self.timer.append(time.perf_counter())
        
        return filtered_points