import numpy as np
import configparser
import json
import cv2
from pypcd4 import PointCloud
from utils import parse_configs, pcd_to_img_map
from ultralytics import YOLO
from PIL import Image
# import open3d as o3d


config = configparser.ConfigParser()
config.read('settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)


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
    
    row_num = int((x_max_border - x_min_border - 1) // PCD_A_W + 1) * 2 - 1
    col_num = int((y_max_border - y_min_border - 1) // PCD_A_H + 1) * 2 - 1
    
    point_areas = get_overlapping_point_areas(points_array, row_num, col_num, x_min_border, y_min_border)

    images = []

    for row in range(row_num):
        for col in range(col_num):
            cur_area_point_arr = np.asarray(point_areas[row][col])
            
            if len(cur_area_point_arr) == 0:
                continue
            
            img_map = pcd_to_img_map(cur_area_point_arr, row, col, x_min_border, y_min_border, z_min, z_max, PCD_A_W / 2, PCD_A_H / 2)
            bevImage = (img_map * 255).astype(np.uint8)
            bevImage = cv2.cvtColor(bevImage, cv2.COLOR_RGB2BGR)
            
            images.append(bevImage)
            # images.append(Image.fromarray(bevImage).convert('RGB'))
            
            # bevImage = cv2.cvtColor(bevImage, cv2.COLOR_BGR2RGB)
            # images.append(bevImage)
        
    return images

def main():
    # TODO: ввод не только 1 файла
    pcd = PointCloud.from_path("training/PCD/041/00.pcd")
    points_array = pcd.numpy(("x", "y", "z", "i"))
    
    with open("training/PCD/024/poses.json") as f:
        json_poses = json.load(f)
        # TODO: json poses под pcd
        position = json_poses[0]['position']
    
    bev_images = get_bev_imgs(points_array, position)
    
    # img = cv2.cvtColor(bev_images[0], cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f"testaaa.jpg", img)
    
    
    model = YOLO('runs/bev_obb_model8/weights/best.pt')
    
    results = model(bev_images[0])
    
    for result in results:
        result.show()
    
    # cv2.imwrite(f"testaaaa.png", bev_images[0])


if __name__ == "__main__":
    main()