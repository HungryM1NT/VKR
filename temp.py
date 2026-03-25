from ultralytics import YOLO

def main():
    model = YOLO('last.pt')

    image_path = './test00.png'

    results = model(image_path)

    for result in results:
        result.show()
        
        # output_path = 'result_image.jpg'
        # result.save(filename=output_path)
        
        # print(f"Готово! Результат сохранен в: {output_path}")

    print(len(results[0]))

if __name__ == "__main__":
    main()