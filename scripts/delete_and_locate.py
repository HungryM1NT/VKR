import configparser
from pypcd4 import PointCloud
from bev_detector import BEV_DETECTOR
from localizer import Localizer
from ultralytics import YOLO
import json
import time
from utils.general import pcd_from_points

config = configparser.ConfigParser()
config.read('scripts/settings.conf')


def main():
    model = YOLO('runs/bev_obb_model14/weights/best.engine', task='obb')
    detector = BEV_DETECTOR(model)

    fold_num = "004"
    pcd_num = "17"
    hd_map_path = f"training/HD_my/{fold_num}.pcd"
    current_pcd_path = f"training/PCD/{fold_num}/{pcd_num}.pcd"
    poses_folder_path = f"training/PCD/{fold_num}/poses.json"
    
    hd_map = PointCloud.from_path(hd_map_path)
    hd_map = hd_map.numpy(("x", "y", "z"))
    
    scan_localizer = Localizer(hd_map)
    
    with open(poses_folder_path) as f:
        json_poses = json.load(f)
        position = json_poses[int(pcd_num)]['position']
    
    for _ in range(2):
        start_clean = time.perf_counter()
        clear_points = detector.detect_and_delete(current_pcd_path, position['x'], position['y'])
        clean_duration = time.perf_counter() - start_clean
        
        clear_points[:, 0] -= position['x']
        clear_points[:, 1] -= position['y']
        clear_points[:, 2] -= position['z']
        
        start_loc = time.perf_counter()
        transform = scan_localizer.process_frame(clear_points)
        loc_duration = time.perf_counter() - start_loc
        
        print(transform)
        print(position)
        print(f"Время очищения (detect_and_delete): {clean_duration:.4f} сек.")
        print(f"Время локализации (process_frame): {loc_duration:.4f} сек.")
    clear_pcd = pcd_from_points(clear_points)
    clear_pcd.save("test.pcd")




if __name__ == "__main__":
    main()
