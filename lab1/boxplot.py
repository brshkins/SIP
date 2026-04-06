import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel("data.xls")

data["Meat production (2022)"] = (
    data["Meat production (2022)"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float)
)

values = data["Meat production (2022)"].dropna()

# логарифмирование данных
log_values = np.log(values)

# фигура
plt.figure(figsize=(10,5))

# Boxplot
plt.subplot(1,2,1)
plt.boxplot(
    log_values,
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen"),
    medianprops=dict(color="red", linewidth=2)
)
plt.title("Boxplot")
plt.ylabel("ln(Meat production)")

# Violin plot
plt.subplot(1,2,2)
plt.violinplot(log_values, showmeans=True)
plt.title("Violin plot")

plt.tight_layout()
plt.show()