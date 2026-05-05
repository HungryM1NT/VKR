import json
from ultralytics import YOLO
from ..bev_detector import BEV_DETECTOR
from ..utils.general import pcd_from_points


model = YOLO('runs/bev_obb_model9/weights/best.engine', task='obb')
detector = BEV_DETECTOR(model)

folder_num = "041"
pcd_num = "00"
pcd_path = f"training/PCD/{folder_num}/{pcd_num}.pcd"
poses_path = f"training/PCD/{folder_num}/poses.json"

with open(poses_path) as f:
    json_poses = json.load(f)
    position = json_poses[int(pcd_num)]['position']
    
clear_points = detector.detect_and_delete(pcd_path, position['x'], position['y'])

clear_pcd = pcd_from_points(clear_points)
clear_pcd.save("done123.pcd")