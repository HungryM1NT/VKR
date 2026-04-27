import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка данных
# Убедитесь, что скрипт лежит в той же папке, что и results.csv
df = pd.read_csv('runs/bev_obb_model8/results.csv')

# 2. Очистка названий колонок от лишних пробелов (YOLO часто их оставляет)
df.columns = df.columns.str.strip()

# 3. Извлечение нужных данных
epochs = df['epoch']
map50_raw = df['metrics/mAP50(B)']

# Строим сглаженную кривую с помощью скользящего среднего (окно 5-10 эпох)
# Это сымитирует "оранжевую" пунктирную линию с оригинального графика
map50_smooth = map50_raw.rolling(window=7, min_periods=1).mean()

# 4. Настройка стиля графика (академический, строгий)
plt.style.use('seaborn-v0_8-whitegrid') # Светлая сетка
plt.figure(figsize=(10, 6), dpi=300)    # Высокое разрешение для презентации

# 5. Отрисовка линий
# Сырые данные (полупрозрачные, тонкие, чтобы не отвлекали)
plt.plot(epochs, map50_raw, color='#8cb8f0', alpha=0.6, linewidth=1.5, label='mAP50 (Сырые данные)')

# Сглаженные данные (яркая, толстая линия - главный фокус)
plt.plot(epochs, map50_smooth, color='#0033a0', linewidth=3, label='mAP50 (Сглаженная)')

# 6. Настройка осей и заголовков
plt.title('Динамика метрики mAP50 в процессе обучения', fontsize=18, pad=15, fontweight='bold', color='#1a1a1a')
plt.xlabel('Эпоха', fontsize=14, labelpad=10)
plt.ylabel('mAP50', fontsize=14, labelpad=10)

# Настройка размера шрифта для делений на осях
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Ограничение оси Y для наглядности (по графику видно, что метрика от ~0.2 до 0.75)
plt.ylim(0.2, 0.8)
plt.xlim(0, max(epochs))

# 7. Легенда
plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

# 8. Сохранение графика
plt.tight_layout()
plt.savefig('map50_presentation.png', format='png', bbox_inches='tight')
plt.show()

print("График успешно сохранен как 'map50_presentation.png'")