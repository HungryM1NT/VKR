import time
import small_gicp
import numpy as np


class Localizer:
    def __init__(self, hd_map, voxel_size=0.2):
        self.voxel_size = voxel_size
        
        self.map_pcd, self.map_tree = small_gicp.preprocess_points(
            hd_map,
            self.voxel_size
        )
        
        self.last_transform = np.eye(4)
    
    def process_frame(self, scan_points):
        start_time = time.time()
        
        scan_xyz = scan_points[:, :3]
        
        scan_pcd, scan_tree = small_gicp.preprocess_points(
            scan_xyz,
            self.voxel_size
        )
        
        result = small_gicp.align(
            self.map_pcd,
            scan_pcd,
            self.map_tree,
            init_T_target_source=self.last_transform,
            max_correspondence_distance=3.0
        )
        
        current_transform = result.T_target_source
        self.last_transform = current_transform
        
        process_time = (time.time() - start_time) * 1000
        # print(f"Время: {process_time} мс | Сходимость: {result.converged}")

        return current_transform