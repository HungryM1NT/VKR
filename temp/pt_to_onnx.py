from ultralytics import YOLO

def export_to_onnx():
    model_path = "/home/ilya/VKR/runs/bev_obb_model8/weights/best.pt" 
    model = YOLO(model_path)

    print("Начинаем экспорт в ONNX...")
    exported_file = model.export(
        format="onnx",
        imgsz=640,
        half=True,
        dynamic=True,
        simplify=True
    )
    
    print(f"Экспорт завершен! Файл сохранен тут: {exported_file}")

if __name__ == "__main__":
    export_to_onnx()