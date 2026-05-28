import os
import cv2
import glob
import yaml

YOLO_DATASET_PATH = 'training/YOLO_data'
BIRDNET_DATASET_PATH = 'training/birdnet_dataset'

def get_classes_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if 'names' in data:
        return [data['names'][k].replace(' ', '_') for k in sorted(data['names'].keys())]
    return []

def convert_yolo_to_dota(yolo_dir, output_dir):
    yaml_path = os.path.join(yolo_dir, 'data.yaml')
    classes = get_classes_from_yaml(yaml_path)
    
    print(f"Найдено классов в data.yaml: {classes}")

    for subset in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'labelTxt'), exist_ok=True)

        img_paths = glob.glob(os.path.join(yolo_dir, subset, 'images', '*.png'))
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            h, w, _ = img.shape
            
            cv2.imwrite(os.path.join(output_dir, subset, 'images', img_name), img)

            label_name = img_name.replace('.png', '.txt')
            yolo_label_path = os.path.join(yolo_dir, subset, 'labels', label_name)
            dota_label_path = os.path.join(output_dir, subset, 'labelTxt', label_name)

            if not os.path.exists(yolo_label_path):
                open(dota_label_path, 'w').close()
                continue

            with open(yolo_label_path, 'r') as f_in, open(dota_label_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) != 9:
                        continue
                    
                    class_idx = int(parts[0])
                    class_name = classes[class_idx]
                    
                    pts = [float(p) for p in parts[1:]]
                    x1, y1 = pts[0] * w, pts[1] * h
                    x2, y2 = pts[2] * w, pts[3] * h
                    x3, y3 = pts[4] * w, pts[5] * h
                    x4, y4 = pts[6] * w, pts[7] * h
                    
                    # DOTA: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                    f_out.write(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f} {class_name} 0\n")

if __name__ == '__main__':
    convert_yolo_to_dota(YOLO_DATASET_PATH, BIRDNET_DATASET_PATH)
    print(f"Новый датасет лежит в: {BIRDNET_DATASET_PATH}")