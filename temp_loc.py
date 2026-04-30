import open3d as o3d
import numpy as np

# 1. Загрузка собранной HD-карты и нового неизвестного скана
print("Загрузка облаков точек...")
global_map = o3d.io.read_point_cloud("global_hd_map_001_without_rot.pcd")
current_scan = o3d.io.read_point_cloud("training/PCD/001/03.pcd")

# Обязательный даунсемплинг для скорости локализации (воксель 20 см)
global_map = global_map.voxel_down_sample(voxel_size=0.2)
current_scan = current_scan.voxel_down_sample(voxel_size=0.2)

# 2. Задаем ПРИМЕРНОЕ положение (Initial Guess)
approx_x = 26.815
approx_y = 28.321
approx_z = -0.110

# Создаем базовую матрицу 4x4 с нашим примерным сдвигом
initial_guess = np.eye(4)
initial_guess[0, 3] = approx_x
initial_guess[1, 3] = approx_y
initial_guess[2, 3] = approx_z
# (Если вы примерно знаете и поворот (heading), его тоже можно вписать в эту матрицу)

# 3. Запускаем локальный ICP
# Указываем радиус поиска. Если погрешность примерных координат около 5 метров, ставим 5.0
threshold = 5.0 

print("Вычисляю точное положение (ICP)...")
reg_p2p = o3d.pipelines.registration.registration_icp(
    current_scan, global_map, threshold, initial_guess,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
)

# 4. Результат
print("\nЛокализация успешна!")
print("Идеальная матрица трансформации (точное положение):")
print(reg_p2p.transformation)

# Выводим оценку точности (Fitness: 1.0 - идеальное совпадение, Inlier RMSE - средняя ошибка)
print(f"Оценка совпадения (Fitness): {reg_p2p.fitness:.4f}")
print(f"Средняя ошибка (RMSE): {reg_p2p.inlier_rmse:.4f} м")

# 5. Визуализация результата
# Покрасим карту в серый, а локализованный скан в красный, чтобы увидеть, как он "встал"
global_map.paint_uniform_color([0.5, 0.5, 0.5])
current_scan.paint_uniform_color([1, 0, 0])

# Применяем найденную матрицу к скану для визуализации на карте
current_scan.transform(reg_p2p.transformation)

o3d.visualization.draw_geometries([global_map, current_scan], window_name="Localization Result")