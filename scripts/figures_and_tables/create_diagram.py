from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def count_classes_and_plot(directory_name):
    class_counts = Counter()
    
    base_dir = Path(directory_name)
    
    txt_files = list(base_dir.rglob('*.txt'))
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if parts:
                        try:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Не удалось прочитать файл {file_path}: {e}")

    classes = [0, 1, 2, 3]
    counts = [class_counts.get(cls, 0) for cls in classes]
    
    temp = counts[1]
    counts[1] = counts[3]
    counts[3] = temp

    print("\n--- Результаты подсчета ---")
    for cls, count in zip(classes, counts):
        print(f"Класс {cls}: {count} объектов")

    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'], edgecolor='black')

    plt.title('Распределение классов в BEV-датасете', fontsize=14, pad=15)
    plt.xlabel('Класс объекта', fontsize=12)
    plt.ylabel('Количество', fontsize=12)
    
    class_names = ['Car', 'Pedestrian', 'Pickup Truck', 'Bus']

    plt.xticks(classes, class_names, fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height + (max(counts) * 0.01 if counts else 0.1),
            str(int(height)), 
            ha='center', 
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("diagram.png", dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    count_classes_and_plot('training/BEV_Dataset/labels')