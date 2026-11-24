import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("iris_cleaned.csv")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
Path("charts").mkdir(exist_ok=True)

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nNumerical summary:\n", df[numeric_cols].describe().round(2))
print("\nSpecies counts:\n", df["species"].value_counts())
print("\nAverage measurements by species:\n", df.groupby("species")[numeric_cols].mean().round(2))
print("\nCorrelation matrix:\n", df.corr(numeric_only=True).round(2))

df["species"].value_counts().plot(kind="bar", title="Number of Records by Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig("charts/species_counts.png", dpi=160)
plt.close()

df.groupby("species")["petal_length_cm"].mean().plot(kind="bar", title="Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average petal length (cm)")
plt.tight_layout()
plt.savefig("charts/average_petal_length.png", dpi=160)
plt.close()

for species, group in df.groupby("species"):
    plt.scatter(group["petal_length_cm"], group["petal_width_cm"], label=species)
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("charts/petal_length_vs_width.png", dpi=160)
plt.close()
print("\nDone. Check the charts folder for graphs.")
