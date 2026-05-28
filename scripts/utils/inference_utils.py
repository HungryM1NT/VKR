import configparser
import numpy as np
from utils.general import parse_configs


config = configparser.ConfigParser()
config.read('scripts/settings.conf')
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

def get_delete_mask_obb(pcd_points, scaled_coords, ego_center=None, ego_size=(4.0, 4.0)):
    xy_points = pcd_points[:, :2]
    
    mask = np.zeros(len(pcd_points), dtype=bool)
    
    for poly in scaled_coords:
            min_x, min_y = np.min(poly, axis=0)
            max_x, max_y = np.max(poly, axis=0)
            
            in_aabb = (xy_points[:, 0] >= min_x) & (xy_points[:, 0] <= max_x) & \
                  (xy_points[:, 1] >= min_y) & (xy_points[:, 1] <= max_y)
            
            if not np.any(in_aabb):
                continue
            
            candidate_points = xy_points[in_aabb]
            
            A, B, C = poly[0], poly[1], poly[2]
            
            AB = B - A
            BC = C - B
            
            AP = candidate_points - A
            BP = candidate_points - B
            
            dot_AP_AB = np.dot(AP, AB)
            dot_AB_AB = np.dot(AB, AB)
            
            dot_BP_BC = np.dot(BP, BC)
            dot_BC_BC = np.dot(BC, BC)
            
            inside_mask = (0 <= dot_AP_AB) & (dot_AP_AB <= dot_AB_AB) & \
                        (0 <= dot_BP_BC) & (dot_BP_BC <= dot_BC_BC)
                        
            mask[in_aabb] |= inside_mask
        
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


def delete_points(pcd_points, scaled_coords, ego_center=None, ego_size=(4.0, 4.0)):
    mask = get_delete_mask_obb(pcd_points, scaled_coords, ego_center, ego_size)
    return pcd_points[mask]


def scale_obbs_percentage(obbs, scale_factor):
    if len(obbs) == 0:
        return obbs
        
    centers = np.mean(obbs, axis=1, keepdims=True) 
    
    scaled_obbs = (obbs - centers) * scale_factor + centers
    
    return scaled_obbs