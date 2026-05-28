from ultralytics import YOLO
import torch

model_path = "runs/bev_obb_model14/weights/best.pt"
model = YOLO(model_path)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

metrics = model.val(data="training/YOLO_data/data.yaml", split="val", batch=1, device="0")

map50 = metrics.box.map50
m_recall = metrics.box.mr

print(f"mAP@50:  {map50:.4f}")
print(f"mRecall: {m_recall:.4f}")

print("\nAP@50 по каждой сущности (классу):")
for i, class_idx in enumerate(metrics.box.ap_class_index):
    class_name = model.names[class_idx]
    ap50_class = metrics.box.class_result(i)[2] 
    print(f" - {class_name}: {ap50_class:.4f}")

speeds = metrics.speed
inference_latency = speeds['inference']
end_to_end_latency = speeds['preprocess'] + speeds['inference'] + speeds['postprocess']

print("\nLatency (при batch=1):")
print(f" - Чистый Inference: {inference_latency:.2f} ms ({(1000/inference_latency):.1f} FPS)")
print(f" - End-to-End: {end_to_end_latency:.2f} ms")

peak_vram_bytes = torch.cuda.max_memory_allocated()
peak_vram_mb = peak_vram_bytes / (1024 * 1024)
print(f"\nПиковое потребление VRAM: {peak_vram_mb:.1f} MB")