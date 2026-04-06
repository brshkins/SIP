import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Загрузка данных из Excel
file_path = '../data.xls'

# Читаем Excel. Если в файле несколько листов, можно добавить sheet_name='Sheet1'
try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    # Если вдруг это все-таки CSV, переименованный в XLS, попробуем так:
    # df = pd.read_csv(file_path, encoding='windows-1251')

# 2. Подготовка данных (Топ-10 стран для наглядности)
# Проверь, чтобы названия колонок в коде совпадали с названиями в твоем файле
column_2022 = 'Meat production (2022)'
column_2020 = 'Meat production (2020)'
column_country = 'Country'

df_top = df.nlargest(10, column_2022)

countries = df_top[column_country]
prod_2020 = df_top[column_2020]
prod_2022 = df_top[column_2022]

# 3. Построение сравнительной гистограммы
x = np.arange(len(countries))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, prod_2020, width, label='2020 год', color='#92a8d1')
rects2 = ax.bar(x + width/2, prod_2022, width, label='2022 год', color='#034f84')

# Оформление
ax.set_ylabel('Объем производства (тонны)')
ax.set_title('Сравнительный анализ производства мяса (2020 vs 2022)')
ax.set_xticks(x)
ax.set_xticklabels(countries, rotation=45, ha='right')
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Сохраняем для отчета
plt.savefig('result_plot.png', dpi=300)
plt.show()