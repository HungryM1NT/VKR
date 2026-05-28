import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

df = pd.read_csv('runs/bev_obb_model14/results.csv')

df.columns = df.columns.str.strip()

plt.figure(figsize=(10, 6))

plt.plot(df['epoch'], df['train/box_loss'], label='Обучение (Box Loss)', color='blue', linewidth=2)
plt.plot(df['epoch'], df['val/box_loss'], label='Валидация (Box Loss)', color='red', linewidth=2, linestyle='--')

plt.xlabel('Эпоха')
plt.ylabel('Значение функции потерь')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('loss_curves.png', dpi=300)
