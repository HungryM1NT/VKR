import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

df = pd.read_csv('runs/bev_obb_model14/results.csv')

df.columns = df.columns.str.strip()

plt.figure(figsize=(10, 6))

plt.plot(df['epoch'], df['metrics/precision(B)'], label='Точность (Precision)', color='#1f77b4', linewidth=2)
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Полнота (Recall)', color='#ff7f0e', linewidth=2)
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='#2ca02c', linewidth=2, linestyle='--')

plt.xlabel('Эпоха')
plt.ylabel('Значение')
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('metrics_dynamics.png', dpi=300)
