import time
import numpy as np
import configparser
from pypcd4 import PointCloud
from numba import njit
from utils.general import parse_configs, get_work_pcd_area
from utils.inference_utils import get_coords_from_bevs, scale_obbs_percentage, delete_points

config = configparser.ConfigParser()
config.read('scripts/settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)


@njit(cache=True)
def fast_build_bev(x_coords, y_coords, z_coords, intensities, bw_doubled, bh_doubled):
    intensity_map = np.zeros((bw_doubled, bh_doubled))
    height_map = np.zeros((bw_doubled, bh_doubled))
    density_map = np.zeros((bw_doubled, bh_doubled))
    count_map = np.zeros((bw_doubled, bh_doubled), dtype=np.int32)
    
    max_z_map = np.full((bw_doubled, bh_doubled), -1.0)

    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]

        if 0 <= x < bw_doubled and 0 <= y < bh_doubled:
            z = z_coords[i]
            
            count_map[x, y] += 1

            if z > max_z_map[x, y]:
                max_z_map[x, y] = z
                height_map[x, y] = z
                intensity_map[x, y] = intensities[i]

    log64 = np.log(64.0)
    for i in range(bw_doubled):
        for j in range(bh_doubled):
            c = count_map[i, j]
            if c > 0:
                density = np.log(c + 1.0) / log64
                if density > 1.0:
                    density = 1.0
                density_map[i, j] = density

    return intensity_map, height_map, density_map


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
    
    def pcd_to_big_img_map(self, pcd_points, x_min_border, y_min_border, z_min, z_max):
        pcd_a_w_doubled = PCD_A_W * 2
        pcd_a_h_doubled = PCD_A_H * 2
        bw_doubled = BW * 2
        bh_doubled = BH * 2

        x_raw = pcd_points[:, 0]
        y_raw = pcd_points[:, 1]
        z_raw = pcd_points[:, 2]
        i_raw = pcd_points[:, 3]

        x_coords = np.int32((x_raw - x_min_border) / pcd_a_w_doubled * bw_doubled)
        y_coords = np.int32((y_raw - y_min_border) / pcd_a_h_doubled * bh_doubled)
        z_coords = (z_raw - z_min) / (z_max - z_min)

        x_coords = np.clip(x_coords, 0, bw_doubled - 1)
        y_coords = np.clip(y_coords, 0, bh_doubled - 1)

        intensity_map, height_map, density_map = fast_build_bev(
            x_coords, y_coords, z_coords, i_raw, bw_doubled, bh_doubled
        )
        
        img_map = np.stack([intensity_map, height_map, density_map], axis=-1)
        return img_map
    
    def get_bev_imgs(self, pcd_points, x_center, y_center):   
        x_min_border = x_center - WM
        x_max_border = x_center + WM

        y_min_border = y_center - HM
        y_max_border = y_center + HM
        
        z_min = np.min(pcd_points[:, 2])
        z_max = np.max(pcd_points[:, 2])
       
        big_img_map = self.pcd_to_big_img_map(pcd_points, x_min_border, y_min_border, z_min, z_max)
        bigBevImage = (big_img_map * 255).astype(np.uint8)

        images = [
            bigBevImage[0:BW, 0:BH],
            bigBevImage[0:BW, BH:BH*2],
            bigBevImage[BW:BW*2, 0:BH],
            bigBevImage[BW:BW*2, BH:BH*2],
        ]
        
        borders = [
            [-WM, -HM],
            [-WM, 0],
            [0, -HM],
            [0, 0]
        ]
            
        return images, borders
    
    def detect_and_delete(self, pcd_path, x_center=0.0, y_center=0.0):
        self.timings = {}
        total_start = time.perf_counter()

        t_start = time.perf_counter()
        pcd_points = self._read_pcd(pcd_path)        
        self.timings['1. Чтение файла PCD'] = time.perf_counter() - t_start
        
        coords = self.detect(pcd_points, x_center, y_center)
        
        t_start = time.perf_counter()
        filtered_points = self.delete(pcd_points, coords, x_center, y_center)
        self.timings['5. Удаление точек'] = time.perf_counter() - t_start
        
        self.timings['ОБЩЕЕ ВРЕМЯ (Очищение)'] = time.perf_counter() - total_start

        # print("\n" + "="*45)
        # print("BEV_DETECTOR:")
        # for step, duration in self.timings.items():
        #     print(f"  {step:<26}: {duration:.4f} сек.")
        # print("="*45 + "\n")
        
        return filtered_points
    
    def detect(self, pcd_points, x_center=0.0, y_center=0.0):
        t_start = time.perf_counter()
        bev_images, borders = self.get_bev_imgs(pcd_points, x_center, y_center)
        self.timings['2. Конвертация PCD в BEV'] = time.perf_counter() - t_start
        
        t_start = time.perf_counter()
        results = []
        for img in bev_images:
            res = self.model(img, verbose=False)
            results.extend(res)
        self.timings['3. Инференс YOLO'] = time.perf_counter() - t_start
    
        t_start = time.perf_counter()
        all_coords = get_coords_from_bevs(results, borders)
        scaled_coords = scale_obbs_percentage(all_coords, scale_factor=1.2)
        self.timings['4. Расчет OBB координат'] = time.perf_counter() - t_start
        
        return scaled_coords
       
    def delete(self, pcd_points, object_coords, x_center=0.0, y_center=0.0):
        filtered_points = delete_points(pcd_points,
                                object_coords,
                                ego_center=(x_center, y_center), 
                                ego_size=(5.0, 5.0))
        return filtered_points