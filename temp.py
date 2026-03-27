from ultralytics import YOLO

def main():
    model = YOLO('runs/bev_obb_model7/weights/best.pt')

    image_path = 'training/YOLO_data/test/images/044_43_00.jpg'

    results = model(image_path)

    for result in results:
        result.show()
        
        # output_path = 'result_image.jpg'
        # result.save(filename=output_path)
        
        # print(f"Готово! Результат сохранен в: {output_path}")

    print(len(results[0]))

if __name__ == "__main__":
    main()