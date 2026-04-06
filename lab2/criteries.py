import pandas as pd
from scipy.stats import shapiro, wilcoxon, spearmanr, kendalltau

ALPHA = 0.05

file_path = "data.xls"
df = pd.read_excel(file_path, engine="xlrd")
df = df[[
    "Country",
    "Meat production (2018)",
    "Meat production (2020)",
    "Meat production (2022)",
    "Milk production (2022)"
]].copy()
for col in [
    "Meat production (2018)",
    "Meat production (2020)",
    "Meat production (2022)",
    "Milk production (2022)"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=[
    "Country",
    "Meat production (2018)",
    "Meat production (2020)",
    "Meat production (2022)",
    "Milk production (2022)"
])
df = df.reset_index(drop=True)
x2022 = df["Meat production (2022)"]
x2020 = df["Meat production (2020)"]
x2018 = df["Meat production (2018)"]
m2022 = df["Milk production (2022)"]

print("Количество стран в анализе:", len(df))
print("\nФрагмент данных:")
print(df.head())

# Критерий Шапиро–Уилка
print("\n")
print("КРИТЕРИЙ ШАПИРО–УИЛКА")

stat_2022, p_2022 = shapiro(x2022)
stat_2020, p_2020 = shapiro(x2020)

print("2022 год:")
print(f"W = {stat_2022:.6f}")
print(f"p-value = {p_2022:.6e}")

if p_2022 < ALPHA:
    print("нулевая гипотеза отвергается: распределение за 2022 год не является нормальным.")
else:
    print("нет оснований отвергать нулевую гипотезу: распределение за 2022 год можно считать нормальным.")

print("\n2020 год:")
print(f"W = {stat_2020:.6f}")
print(f"p-value = {p_2020:.6e}")

if p_2020 < ALPHA:
    print("нулевая гипотеза отвергается: распределение за 2020 год не является нормальным.")
else:
    print("нет оснований отвергать нулевую гипотезу: распределение за 2020 год можно считать нормальным.")

# Критерий Вилкоксона
print("\n")
print("КРИТЕРИЙ ВИЛКОКСОНА")

stat_w, p_w = wilcoxon(x2020, x2022, alternative="two-sided")

print(f"\nW = {stat_w:.6f}")
print(f"p-value = {p_w:.6e}")

if p_w < ALPHA:
    print("нулевая гипотеза отвергается: между значениями за 2020 и 2022 годы есть статистически значимые различия.")
else:
    print("нет оснований отвергать нулевую гипотезу: статистически значимые различия между 2020 и 2022 годами не обнаружены.")

# Критерий Спирмена
print("\n")
print("КРИТЕРИЙ СПИРМЕНА")

corr_s, p_s = spearmanr(x2020, x2022)

print(f"\nr_s = {corr_s:.6f}")
print(f"p-value = {p_s:.6e}")

if p_s < ALPHA:
    print("нулевая гипотеза отвергается: между значениями за 2020 и 2022 годы существует статистически значимая связь.")
else:
    print("нет оснований отвергать нулевую гипотезу: статистически значимая связь между значениями не обнаружена.")

# Критерий Кендалла
tau, p_k = kendalltau(x2020, x2022)

print("\n")
print("КРИТЕРИЙ КЕНДАЛЛА")
print("tau =", tau)
print("p-value =", p_k)

if p_k < ALPHA:
    print("нулевая гипотеза отвергается: существует статистически значимая связь.")
else:
    print("нет оснований отвергать нулевую гипотезу.")


# ВИЛКОКСОН (динамика)
print("\n=== Вилкоксон 2018 vs 2020 ===")
w1, p1 = wilcoxon(x2018, x2020)
print("W =", w1, "p =", p1)

print("\n=== Вилкоксон 2020 vs 2022 ===")
w2, p2 = wilcoxon(x2020, x2022)
print("W =", w2, "p =", p2)

# СПИРМЕН (структура)
print("\n=== Спирмен 2018 vs 2020 ===")
s1, ps1 = spearmanr(x2018, x2020)
print("r =", s1, "p =", ps1)

print("\n=== Спирмен 2020 vs 2022 ===")
s2, ps2 = spearmanr(x2020, x2022)
print("r =", s2, "p =", ps2)

# анализ разных показателей
print("\n")
print("АНАЛИЗ РАЗНЫХ ПОКАЗАТЕЛЕЙ (МЯСО vs МОЛОКО)")

milk = df["Milk production (2022)"]

# Спирмен
r_milk, p_milk = spearmanr(x2022, milk)
print("\nСпирмен (meat vs milk):")
print("r =", r_milk, "p =", p_milk)

# Кендалл
tau_milk, p_k_milk = kendalltau(x2022, milk)
print("\nКендалл (meat vs milk):")
print("tau =", tau_milk, "p =", p_k_milk)

if p_milk < ALPHA:
    print("Есть статистически значимая связь между производством мяса и молока.")
else:
    print("Статистически значимая связь не обнаружена.")