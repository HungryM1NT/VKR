import time
import torch
import os
from ultralytics import YOLO

engine_path = 'runs/bev_obb_model14/weights/best.engine' 
data_yaml = 'training/YOLO_data/data.yaml'
test_img = 'training/YOLO_data/validation/images/011_00_00.png'

model = YOLO(engine_path, task='obb')

metrics = model.val(data=data_yaml, device='0', imgsz=640, split='val', batch=4)

map50 = metrics.box.map50
m_recall = metrics.box.mr
classes = metrics.names

print(f"mAP@0.5: {map50:.4f}")
print(f"mRecall: {m_recall:.4f}")

print("\nAP@0.5 по классам:")
for i, class_name in enumerate(classes.values()):
    ap50_class = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0.0
    print(f" - {class_name}: {ap50_class:.4f}")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

for _ in range(10):
    _ = model(test_img, verbose=False)

num_runs = 100
torch.cuda.synchronize()
start_time = time.perf_counter()

for _ in range(num_runs):
    _ = model(test_img, verbose=False)

torch.cuda.synchronize()
end_time = time.perf_counter()

avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
fps = 1000 / avg_latency_ms

peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

print(f"Latency (End-to-End): {avg_latency_ms:.2f} ms")
print(f"FPS:                  {fps:.1f}")
print(f"Пиковое VRAM:         {peak_vram_mb:.1f} MB")
