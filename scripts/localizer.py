import numpy as np
import small_gicp
import time
from pypcd4 import PointCloud


class RealTimeLocalizer:
    def __init__(self, global_map_points, voxel_size=0.2):
        self.voxel_size = voxel_size
        
        self.map_pc, self.map_tree = small_gicp.preprocess_points(
            points=global_map_points, 
            downsampling_resolution=self.voxel_size
        )
        print("Карта успешно загружена в память.")
        
        self.last_transform = np.eye(4)

    def process_frame(self, scan_xyzi, external_initial_guess=None):
        start_time = time.time()

        scan_xyz = scan_xyzi[:, :3]
        scan_xyz[:, 0] -= 5.0
        scan_xyz[:, 1] -= 3.0

        scan_pc, scan_tree = small_gicp.preprocess_points(
            points=scan_xyz, 
            downsampling_resolution=self.voxel_size
        )

        guess = external_initial_guess if external_initial_guess is not None else self.last_transform
        
        result = small_gicp.align(
            self.map_pc, 
            scan_pc, 
            self.map_tree, 
            init_T_target_source=guess,
            max_correspondence_distance=3.0
        )
        
        current_transform = result.T_target_source
        self.last_transform = current_transform
        
        calc_time = (time.time() - start_time) * 1000
        print(f"Кадр обработан за {calc_time:.2f} мс | Сходимость: {result.converged}")

        return current_transform

global_map = PointCloud.from_path("training/HD_perfect/001.pcd")
global_map = global_map.numpy(("x", "y", "z"))
# dummy_global_map = np.random.rand(100000, 3) * 100 
localizer = RealTimeLocalizer(global_map, voxel_size=0.2)

files = ["training/PCD_perfect/001/00.pcd",
         "training/PCD_perfect/001/01.pcd",
         "training/PCD_perfect/001/02.pcd",
         "training/PCD_perfect/001/03.pcd",
         "training/PCD_perfect/001/04.pcd",
         "training/PCD_perfect/001/05.pcd",
         "training/PCD_perfect/001/06.pcd",
         "training/PCD_perfect/001/07.pcd",
         "training/PCD_perfect/001/08.pcd",
         "training/PCD_perfect/001/09.pcd"]

for i in range(len(files)):
    # incoming_scan_xyzi = np.random.rand(15000, 4) * 10
    scan = PointCloud.from_path(files[i])
    scan = scan.numpy(("x", "y", "z"))
    pose = localizer.process_frame(scan)
    # print(pose)