from ultralytics import YOLO

model_path = 'runs/bev_obb_model14/weights/best.pt' 

model = YOLO(model_path)

model.export(
    format='engine',
    device='0',
    imgsz=640,
    half=True,
    workspace=4,
    simplify=True
)

print("\nФайл .engine сохранен в той же папке, где лежал .pt")