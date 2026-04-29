from bev_detector import BEV_DETECTOR
from ultralytics import YOLO
import json
import numpy as np
from pypcd4 import PointCloud


model = YOLO('runs/bev_obb_model9/weights/best.engine', task='obb')
detector = BEV_DETECTOR(model)
TYPES = (np.float32, np.float32, np.float32, np.float32)

pcd_path = "training/PCD/041/00.pcd"
with open("training/PCD/041/poses.json") as f:
    json_poses = json.load(f)
    position = json_poses[0]['position']
    
clear_points = detector.detect_and_delete(pcd_path, position['x'], position['y'])

clear_pcd = PointCloud.from_points(clear_points, ['x', 'y', 'z', 'i'], TYPES)
clear_pcd.save("done123.pcd")