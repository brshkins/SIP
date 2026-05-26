import os
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, spearmanr

warnings.filterwarnings("ignore")

CSV_PATH = "answers.csv"      # имя файла с ответами Google Forms
OUT_DIR = "figures_lr3"              # папка для сохранения рисунков
FORM_SCREENSHOT = "описание_опроса.png"  # скрин начала Google Forms, если нужно сделать рис. 1

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BANKS = ["Сбер", "Т-Банк", "АльфаБанк", "ВТБ", "БСПБ"]

CRITERIA = {
    "Предпочтение / частота использования банка": "Какой банковский сервис вы используете чаще всего?",
    "Оценка банковского приложения": "Оцените приложение",
    "Качество работы при отсутствии интернета": "Качество работы при отсутствии интернета",
    "Выгода условий и кэшбэка": "Выгода условий и кэшбэка",
    "Качество поддержки клиентов": "Качество поддержки клиентов",
    "Скорость работы сервиса": "Скорость работы сервиса",
    "Понятность интерфейса": "Понятность интерфейса",
    "Доверие к банку": "Доверие к банку",
}

SHORT_CRITERIA = {
    "Предпочтение / частота использования банка": "Предпочтение",
    "Оценка банковского приложения": "Приложение",
    "Качество работы при отсутствии интернета": "Оффлайн",
    "Выгода условий и кэшбэка": "Кэшбэк",
    "Качество поддержки клиентов": "Поддержка",
    "Скорость работы сервиса": "Скорость",
    "Понятность интерфейса": "Интерфейс",
    "Доверие к банку": "Доверие",
}

def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def find_col(df: pd.DataFrame, question_start: str, bank: str | None = None) -> str:

    candidates = [c for c in df.columns if c.strip().startswith(question_start)]
    if bank is None:
        if not candidates:
            raise KeyError(f"Не найдена колонка: {question_start}")
        return candidates[0]

    # точный поиск по банку
    for c in candidates:
        if f"[{bank}]" in c or bank in c:
            return c

    # специальный случай: в CSV может быть 'Сбе' вместо 'Сбер'
    if bank == "Сбер":
        for c in candidates:
            if "[Сбе" in c or "Сбе]" in c:
                return c

    raise KeyError(f"Не найдена колонка для вопроса '{question_start}' и банка '{bank}'")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

df = pd.read_csv(CSV_PATH)

sex_col = "Ваш пол"
age_col = "Ваш возраст"
group_col = "Группа"
freq_col = "Как часто Вы пользуетесь банковскими приложениями?"
apps_col = "Сколько банковских приложений Вы используете регулярно?"


mean_scores = pd.DataFrame(index=list(CRITERIA.keys()), columns=BANKS, dtype=float)

for crit_name, q_start in CRITERIA.items():
    for bank in BANKS:
        col = find_col(df, q_start, bank)
        mean_scores.loc[crit_name, bank] = to_num(df[col]).mean()

integral_scores = mean_scores.mean(axis=0)

expert_integral = pd.DataFrame(index=df.index, columns=BANKS, dtype=float)
for bank in BANKS:
    cols = [find_col(df, q_start, bank) for q_start in CRITERIA.values()]
    expert_integral[bank] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

if os.path.exists(FORM_SCREENSHOT):
    img = plt.imread(FORM_SCREENSHOT)
    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.axis("off")
    savefig("01_описание_опроса.png")
else:
    print("Рисунок 1 пропущен: скрин Google Forms нужно сохранить как 'описание_опроса.png'.")

sex_counts = df[sex_col].value_counts()
plt.figure(figsize=(6, 5))
plt.pie(
    sex_counts.values,
    labels=[f"{idx}: {val}" for idx, val in sex_counts.items()],
    autopct="%1.1f%%",
    startangle=0,
)
plt.title("Распределение респондентов по полу")
savefig("02_распределение_по_полу.png")

group_counts = df[group_col].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(group_counts.index.astype(str), group_counts.values)
plt.title("Распределение по учебным группам")
plt.xlabel("Группа")
plt.ylabel("Число ответов")
savefig("03_распределение_по_группам.png")

freq_counts = to_num(df[freq_col]).value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(freq_counts.index.astype(str), freq_counts.values)
plt.title("Частота использования банковских приложений")
plt.xlabel("Оценка 1–5")
plt.ylabel("Число ответов")
savefig("04_частота_использования.png")


apps_counts = to_num(df[apps_col]).value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(apps_counts.index.astype(str), apps_counts.values)
plt.title("Число регулярно используемых приложений")
plt.xlabel("Количество приложений")
plt.ylabel("Число ответов")
savefig("05_число_приложений.png")

plt.figure(figsize=(7, 4))
bars = plt.bar(integral_scores.index, integral_scores.values)
plt.title("Интегральная оценка банков")
plt.xlabel("Банк")
plt.ylabel("Средний балл")
plt.ylim(0, 5)
for bar, val in zip(bars, integral_scores.values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f}", ha="center")
savefig("06_интегральная_оценка_банков.png")

heat = mean_scores.T  # банки × критерии
plt.figure(figsize=(11, 4.8))
im = plt.imshow(heat.values, aspect="auto")
plt.colorbar(im, label="Средний балл")
plt.title("Средние оценки банков по критериям")
plt.xticks(range(len(heat.columns)), [SHORT_CRITERIA[c] for c in heat.columns], rotation=35, ha="right")
plt.yticks(range(len(heat.index)), heat.index)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        plt.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
savefig("07_тепловая_карта_средних_оценок.png")

rows = []
for expert_idx in df.index:
    for bank in BANKS:
        row = {"Эксперт": expert_idx + 1, "Банк": bank}
        for crit_name, q_start in CRITERIA.items():
            row[crit_name] = float(df.loc[expert_idx, find_col(df, q_start, bank)])
        rows.append(row)
obj = pd.DataFrame(rows)
X = obj[list(CRITERIA.keys())].to_numpy(dtype=float)
Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

# PCA через SVD
U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
scores = U * S
# собственные значения корреляционной матрицы
n_obs = Xz.shape[0]
eigenvalues = (S ** 2) / (n_obs - 1)
loadings = Vt.T * np.sqrt(eigenvalues)

if loadings[:, 0].sum() < 0:
    scores[:, 0] *= -1
    loadings[:, 0] *= -1
# F2 должен быть положительно связан с оффлайн-доступом.
offline_idx = list(CRITERIA.keys()).index("Качество работы при отсутствии интернета")
if loadings[offline_idx, 1] < 0:
    scores[:, 1] *= -1
    loadings[:, 1] *= -1

obj["F1"] = scores[:, 0]
obj["F2"] = scores[:, 1]
bank_factor_scores = obj.groupby("Банк")[["F1", "F2"]].mean().loc[BANKS]

plt.figure(figsize=(7, 5))
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.scatter(bank_factor_scores["F1"], bank_factor_scores["F2"], s=60)
for bank in BANKS:
    x, y = bank_factor_scores.loc[bank, ["F1", "F2"]]
    plt.text(x + 0.03, y + 0.01, bank)
plt.title("Положение банков в пространстве факторов F1–F2")
plt.xlabel("F1: общая удовлетворенность")
plt.ylabel("F2: автономность / работа без интернета")
savefig("08_положение_банков_F1_F2.png")

chi2_values = []
p_values = []
w_values = []
labels = []

for crit_name, q_start in CRITERIA.items():
    arrays = [to_num(df[find_col(df, q_start, bank)]).to_numpy() for bank in BANKS]
    chi2, p = friedmanchisquare(*arrays)
    w = chi2 / (len(df) * (len(BANKS) - 1))
    labels.append(SHORT_CRITERIA[crit_name])
    chi2_values.append(chi2)
    p_values.append(p)
    w_values.append(w)

# интегральная оценка
chi2_int, p_int = friedmanchisquare(*[expert_integral[bank].to_numpy() for bank in BANKS])
w_int = chi2_int / (len(df) * (len(BANKS) - 1))
labels.append("Интегр.")
chi2_values.append(chi2_int)
p_values.append(p_int)
w_values.append(w_int)

plt.figure(figsize=(8, 4))
plt.bar(labels, w_values)
plt.title("Коэффициент конкордации Кендалла W")
plt.xlabel("Критерий")
plt.ylabel("W")
plt.xticks(rotation=35, ha="right")
plt.ylim(0, max(w_values) * 1.25)
savefig("09_коэффициент_конкордации_W.png")

expert_mean_y = expert_integral.mean(axis=1).to_numpy(dtype=float)
sex_code = df[sex_col].map({"Женский": 0, "Мужской": 1}).astype(float).to_numpy()
group_map = {4414: 1, 4415: 2, 4416: 3, 4417: 4, "4414": 1, "4415": 2, "4416": 3, "4417": 4}
group_code = df[group_col].map(group_map).astype(float).to_numpy()
freq_code = to_num(df[freq_col]).to_numpy(dtype=float)
apps_code = to_num(df[apps_col]).to_numpy(dtype=float)

Xreg = np.column_stack([np.ones(len(df)), sex_code, group_code, freq_code, apps_code])
beta = np.linalg.lstsq(Xreg, expert_mean_y, rcond=None)[0]
coef_labels = ["Пол\n(мужской=1)", "Группа", "Частота\nиспользования", "Число\nприложений"]
coef_values = beta[1:]

plt.figure(figsize=(7, 4))
plt.axhline(0, linewidth=0.8)
plt.bar(coef_labels, coef_values)
plt.title("Коэффициенты регрессионной модели")
plt.ylabel("Коэффициент")
savefig("10_коэффициенты_регрессии.png")

group_consensus = expert_integral.mean(axis=0).to_numpy(dtype=float)
rhos = []
for _, row in expert_integral.iterrows():
    rho, _ = spearmanr(row.to_numpy(dtype=float), group_consensus)
    rhos.append(rho)
rhos = np.array(rhos, dtype=float)
weak_mask = rhos < 0.20
filtered_integral = expert_integral.loc[~weak_mask]

chi2_before, p_before = friedmanchisquare(*[expert_integral[bank].to_numpy() for bank in BANKS])
w_before = chi2_before / (len(expert_integral) * (len(BANKS) - 1))
chi2_after, p_after = friedmanchisquare(*[filtered_integral[bank].to_numpy() for bank in BANKS])
w_after = chi2_after / (len(filtered_integral) * (len(BANKS) - 1))

plt.figure(figsize=(6, 4))
vals = [w_before, w_after]
bars = plt.bar(["До удаления", "После удаления"], vals)
plt.title("Пересчет согласованности группы")
plt.ylabel("W по интегральной оценке")
plt.ylim(0, max(vals) * 1.25)
for bar, val in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center")
savefig("11_пересчет_согласованности.png")

summary = pd.DataFrame({
    "Критерий": labels,
    "chi2": chi2_values,
    "p_value": p_values,
    "W": w_values,
})
summary.to_csv(os.path.join(OUT_DIR, "таблица_Фридман_W.csv"), index=False, encoding="utf-8-sig")
mean_scores.T.assign(**{"Интегральная оценка": integral_scores}).to_csv(
    os.path.join(OUT_DIR, "средние_оценки_банков.csv"), encoding="utf-8-sig"
)

print("\nГотово. Все рисунки сохранены в папке:", OUT_DIR)
