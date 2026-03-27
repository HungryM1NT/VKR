from ultralytics import YOLO
import configparser


config = configparser.ConfigParser()
config.read('settings.conf')
YOLO_DATASET_PATH = config['PATHS']['YOLO_DATASET_PATH']


def main():
    yaml_path = f"{YOLO_DATASET_PATH}/data.yaml"

    model = YOLO("yolo26n-obb.pt") 

    results = model.train(
        data=yaml_path,
        project="/home/ilya/VKR/runs",
        epochs=int(config['YOLO_HYPERPARAMS']['EPOCHS']),
        imgsz=int(config['CONSTANTS']['BEV_WIDTH']),
        batch=int(config['YOLO_HYPERPARAMS']['BATCH']),
        device=0,
        name="bev_obb_model",
        workers=int(config['YOLO_HYPERPARAMS']['WORKERS']),
        
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        perspective=0.0,
        scale=0.5,
    )
    
if __name__ == "__main__":
    main()