import numpy as np
import configparser
import json
import cv2
import time
from pypcd4 import PointCloud
from utils import parse_configs, pcd_to_img_map, get_point_areas
from ultralytics import YOLO
from PIL import Image
from delete_utils import get_coords_from_bevs, delete_points, scale_obbs_percentage


config = configparser.ConfigParser()
config.read('settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)

TYPES = (np.float32, np.float32, np.float32, np.float32)


def get_overlapping_point_areas(points_array, row_num, col_num, x_min_border, y_min_border):
    point_areas = [[[] for _ in range(col_num)] for _ in range(row_num)]
    
    step_x = PCD_A_W / 2
    step_y = PCD_A_H / 2

    for point in points_array:
        px, py = point[0], point[1]
        
        for x_idx in range(row_num):
            area_x_min = x_min_border + x_idx * step_x
            area_x_max = area_x_min + PCD_A_W
            
            if area_x_min <= px < area_x_max:
                for y_idx in range(col_num):
                    area_y_min = y_min_border + y_idx * step_y
                    area_y_max = area_y_min + PCD_A_H
                    
                    if area_y_min <= py < area_y_max:
                        point_areas[x_idx][y_idx].append(point)
    
    return point_areas

def get_bev_imgs(points_array, position):
    x_center = position['x']
    y_center = position['y']
        
    x_min_border = x_center - WM
    x_max_border = x_center + WM

    y_min_border = y_center - HM
    y_max_border = y_center + HM
    
    z_min = np.min(points_array[:, 2])
    z_max = np.max(points_array[:, 2])
    
    row_num = int((x_max_border - x_min_border - 1) // PCD_A_W + 1)
    col_num = int((y_max_border - y_min_border - 1) // PCD_A_H + 1)
    
    # point_areas = get_overlapping_point_areas(points_array, row_num, col_num, x_min_border, y_min_border)
    point_areas = get_point_areas(points_array, row_num, col_num, x_min_border, y_min_border)

    # step_x, step_y = PCD_A_W / 2, PCD_A_H / 2
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
            bevImage = cv2.cvtColor(bevImage, cv2.COLOR_RGB2BGR)
            
            images.append(bevImage)
            
            cur_x_min_border = x_min_border + row * step_x
            cur_y_min_border = y_min_border + col * step_y
            
            borders.append((cur_x_min_border, cur_y_min_border))
            # images.append(Image.fromarray(bevImage).convert('RGB'))
            
            # bevImage = cv2.cvtColor(bevImage, cv2.COLOR_BGR2RGB)
            # images.append(bevImage)
        
    return images, borders

def main():
    model = YOLO('runs/bev_obb_model9/weights/best.engine', task='obb')
    print("Прогрев модели...")
    dummy_img = [np.zeros((BH, BW, 3), dtype=np.uint8)]
    model(dummy_img, verbose=False)
    
    # TODO: ввод не только 1 файла
    start_total_time = time.perf_counter()
    pcd = PointCloud.from_path("training/PCD/041/00.pcd")
    points_array = pcd.numpy(("x", "y", "z", "i"))
    
    with open("training/PCD/041/poses.json") as f:
        json_poses = json.load(f)
        # TODO: json poses под pcd
        position = json_poses[0]['position']
    
    time_after_load = time.perf_counter()
    bev_images, borders = get_bev_imgs(points_array, position)
    
    # img = cv2.cvtColor(bev_images[0], cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f"testaaa.jpg", img)
    
    time_after_prep = time.perf_counter()
    
    # results = model(bev_images, half=True, verbose=False)
    # results = model(bev_images, verbose=False)
    
    results = []
    for img in bev_images:
        res = model(img, verbose=False)
        results.extend(res)
        
    # for result in results:
    #     result.show()
    time_after_inference = time.perf_counter()
    # print(borders[0])
    
    
    all_coords = get_coords_from_bevs(results, borders)
    scaled_coords = scale_obbs_percentage(all_coords, scale_factor=1.2)
    
    filtered_points = delete_points(points_array,
                                    scaled_coords,
                                    ego_center=(position['x'], position['y']), 
                                    ego_size=(5.0, 5.0))
    
    clear_pcd = PointCloud.from_points(filtered_points, PCD_COL, TYPES)
    
    
    clear_pcd.save("done1.pcd")
    
    # for i in range(len(bev_images)):
    #     cv2.imwrite(f"aaaaa{i}.png", bev_images[i])
    
    end_total_time = time.perf_counter()
    
    # === ВЫВОД РЕЗУЛЬТАТОВ ===
    total_time = end_total_time - start_total_time
    print(f"\n--- Тайминг обработки скана ---")
    print(f"Чтение файла:       {(time_after_load - start_total_time) * 1000:.1f} мс")
    print(f"PCD -> BEV:         {(time_after_prep - time_after_load) * 1000:.1f} мс")
    print(f"YOLO инференс:      {(time_after_inference - time_after_prep) * 1000:.1f} мс")
    print(f"Постпроцессинг:     {(end_total_time - time_after_inference) * 1000:.1f} мс")
    print(f"Итоговое время:     {total_time * 1000:.1f} мс (~{1 / total_time:.1f} FPS)\n")


if __name__ == "__main__":
    main()