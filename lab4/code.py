from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

years = np.array([
    2000, 2001, 2002, 2003, 2004,
    2005, 2006, 2007, 2008, 2009,
    2010, 2011, 2012, 2013, 2014,
    2015, 2016, 2017, 2018, 2019,
    2020, 2021, 2022, 2023, 2024
])

meat = np.array([
    231.7, 235.3, 242.6, 248.3, 257.0,
    264.3, 270.8, 278.6, 286.6, 288.2,
    296.1, 301.6, 306.3, 313.0, 318.2,
    323.2, 327.7, 332.5, 342.5, 337.8,
    338.5, 354.4, 360.5, 366.7, 374.0
])

df = pd.DataFrame({
    "Год": years,
    "Производство мяса, млн т": meat
})

df["Абсолютный прирост цепной, млн т"] = df["Производство мяса, млн т"].diff()
df["Темп прироста цепной, %"] = (
    df["Производство мяса, млн т"].pct_change() * 100
)

# Номер периода в ряду: t = 1, 2, ..., 25
t = np.arange(1, len(df) + 1)

series = pd.Series(meat, index=years)

ma3 = series.rolling(window=3, center=True).mean()
ma5 = series.rolling(window=5, center=True).mean()

ma3_for_plot = ma3.bfill().ffill()
ma5_for_plot = ma5.bfill().ffill()

# Экспоненциальное сглаживание с alpha = 0.3
exp_smooth = series.ewm(alpha=0.3, adjust=False).mean()

# Отклонения от сглаженной линии MA(5)
deviation_from_ma5 = series - ma5_for_plot


# Автокорреляция лага k.
def autocorrelation(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    centered = x - x.mean()

    if lag == 0:
        return 1.0

    numerator = np.sum(centered[lag:] * centered[:-lag])
    denominator = np.sum(centered ** 2)
    return numerator / denominator


lags = np.arange(0, 11)
acf_values = np.array([autocorrelation(meat, lag) for lag in lags])
acf_bound = 1.96 / np.sqrt(len(meat))

linear_coef = np.polyfit(t, meat, deg=1)
quadratic_coef = np.polyfit(t, meat, deg=2)
cubic_coef = np.polyfit(t, meat, deg=3)

linear_trend = np.polyval(linear_coef, t)
quadratic_trend = np.polyval(quadratic_coef, t)
cubic_trend = np.polyval(cubic_coef, t)

# Экспоненциальная модель: y_hat = A * exp(b * t)
exp_coef = np.polyfit(t, np.log(meat), deg=1)
b = exp_coef[0]
ln_A = exp_coef[1]
A = np.exp(ln_A)
exponential_trend = A * np.exp(b * t)

# Остатки выбранной квадратичной модели.
quadratic_residuals = meat - quadratic_trend

OUT_DIR = Path("plots_pz4")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 125,
    "savefig.dpi": 125,
})


def save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close()

# Рисунок 1. Динамика временного ряда
plt.figure(figsize=(13, 7.2))
plt.plot(years, meat, marker="o")
plt.title("Динамика мирового производства мяса, 2000-2024 гг.")
plt.xlabel("Год")
plt.ylabel("Производство мяса, млн тонн")
plt.grid(True, alpha=0.3)
save_current_figure("01_dynamics.png")


# Рисунок 2. Цепной абсолютный прирост
plt.figure(figsize=(13, 7.2))
plt.bar(
    df["Год"].iloc[1:],
    df["Абсолютный прирост цепной, млн т"].iloc[1:]
)
plt.axhline(0, linewidth=1)
plt.title("Цепной абсолютный прирост производства мяса")
plt.xlabel("Год")
plt.ylabel("Прирост к предыдущему году, млн тонн")
plt.grid(axis="y", alpha=0.3)
save_current_figure("02_chain_growth.png")


# Рисунок 3. Коррелограмма временного ряда
plt.figure(figsize=(13, 7.2))
markerline, stemlines, baseline = plt.stem(lags, acf_values)

# Базовая линия у stem по умолчанию красная — как в отчете.
plt.setp(markerline, markersize=6)
plt.setp(stemlines, linewidth=1.5)
plt.setp(baseline, linewidth=1.0)

plt.axhline(acf_bound, linestyle="--", linewidth=1)
plt.axhline(-acf_bound, linestyle="--", linewidth=1)

plt.title("Коррелограмма временного ряда")
plt.xlabel("Лаг k")
plt.ylabel("Оценка автокорреляции r_k")
plt.xticks(lags)
plt.ylim(-0.5, 1.1)
plt.grid(True, alpha=0.3)
save_current_figure("03_correlogram.png")


# Рисунок 4. Выделение тренд-циклической составляющей MA(5)
plt.figure(figsize=(13, 7.2))
plt.plot(years, meat, marker="o", label="Фактический ряд")
plt.plot(years, ma5_for_plot.values, label="Тренд-цикл MA(5)")
plt.title("Выделение тренд-циклической составляющей методом MA(5)")
plt.xlabel("Год")
plt.ylabel("Производство мяса, млн тонн")
plt.grid(True, alpha=0.3)
plt.legend()
save_current_figure("04_ma5_trend_cycle.png")


# Рисунок 5. Отклонения фактического ряда от MA(5)
plt.figure(figsize=(13, 6.85))
plt.bar(years, deviation_from_ma5.values)
plt.axhline(0, linewidth=1)
plt.title("Отклонение фактического ряда от тренд-цикла MA(5)")
plt.xlabel("Год")
plt.ylabel("Отклонение, млн тонн")
plt.grid(axis="y", alpha=0.3)
save_current_figure("05_deviation_from_ma5.png")


# Рисунок 6. Сравнение методов сглаживания
plt.figure(figsize=(13, 7.2))
plt.plot(years, meat, marker="o", label="Факт")
plt.plot(years, ma3_for_plot.values, label="MA(3)")
plt.plot(years, ma5_for_plot.values, label="MA(5)")
plt.plot(years, exp_smooth.values, label="Эксп. сглаживание α=0.3")
plt.title("Сравнение методов сглаживания")
plt.xlabel("Год")
plt.ylabel("Производство мяса, млн тонн")
plt.grid(True, alpha=0.3)
plt.legend()
save_current_figure("06_smoothing_compare.png")


# Рисунок 7. Аппроксимация ряда моделями тренда
plt.figure(figsize=(13, 7.2))
plt.plot(years, meat, marker="o", label="Факт")
plt.plot(years, linear_trend, label="Линейная")
plt.plot(years, quadratic_trend, label="Квадратичная")
plt.plot(years, exponential_trend, label="Экспоненциальная")
plt.title("Аппроксимация ряда моделями тренда")
plt.xlabel("Год")
plt.ylabel("Производство мяса, млн тонн")
plt.grid(True, alpha=0.3)
plt.legend()
save_current_figure("07_trend_models.png")


# Рисунок 8. Остаточная компонента для квадратичной модели
plt.figure(figsize=(13, 6.85))
plt.bar(years, quadratic_residuals)
plt.axhline(0, linewidth=1)
plt.title("Остаточная компонента для выбранной квадратичной модели")
plt.xlabel("Год")
plt.ylabel("Остаток, млн тонн")
plt.grid(axis="y", alpha=0.3)
save_current_figure("08_quadratic_residuals.png")


# Рисунок 9. Гистограмма остаточной компоненты
plt.figure(figsize=(13, 6.85))
plt.hist(quadratic_residuals, bins=7, edgecolor="black")
plt.title("Гистограмма остаточной компоненты")
plt.xlabel("Остаток, млн тонн")
plt.ylabel("Частота")
plt.grid(axis="y", alpha=0.3)
save_current_figure("09_residual_histogram.png")


print(f"Готово. Все графики сохранены в папку: {OUT_DIR.resolve()}")
