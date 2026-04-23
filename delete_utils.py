import configparser
import numpy as np
from utils import parse_configs
from matplotlib.path import Path


config = configparser.ConfigParser()
config.read('settings.conf')
CUB_COL, PCD_COL, PCD_A_W, PCD_A_H, BW, BH, WM, HM = parse_configs(config)


def get_coords_from_bevs(yolo_results, borders):
    all_coords = []
    
    for r, border in zip(yolo_results, borders):
        if r.obb is not None:
            x_min_border, y_min_border = border

            corners = r.obb.xyxyxyxy.cpu().numpy().copy()
            
            pixel_x = corners[:, :, 0].copy()
            pixel_y = corners[:, :, 1].copy()
            
            corners[:, :, 0] = pixel_y * (PCD_A_W / BW) + x_min_border 
            corners[:, :, 1] = pixel_x * (PCD_A_H / BH) + y_min_border
            
            for corner in corners:
                all_coords.append(corner)
                
    return np.array(all_coords)


def get_delete_mask_obb(pcd_points, obb_polygons):
    xy_points = pcd_points[:, :2]
    
    mask = np.full((len(pcd_points)), False)
    
    for poly in obb_polygons:
        poly_path = Path(poly)
        inside_mask = poly_path.contains_points(xy_points)
        
        mask = np.logical_or(mask, inside_mask)
    
    return np.invert(mask)

def delete_points(pcd_points, obb_polygons):
    mask = get_delete_mask_obb(pcd_points, obb_polygons)
    return pcd_points[mask]

def scale_obbs_percentage(obbs, scale_factor):
    if len(obbs) == 0:
        return obbs
        
    centers = np.mean(obbs, axis=1, keepdims=True) 
    
    scaled_obbs = (obbs - centers) * scale_factor + centers
    
    return scaled_obbs