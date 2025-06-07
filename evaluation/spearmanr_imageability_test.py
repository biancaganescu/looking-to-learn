import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, kruskal, f_oneway
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./gate_soft_per_token/word_gate_imageability.csv")

df_filtered = df[df["Imageability"] > 0]

corr, p_corr = spearmanr(df_filtered["Imageability"], df_filtered["gate"])
print(f"Spearman correlation: {corr:.3f}, P-value: {p_corr:.3f}")

imag_bins = [100, 342, 450, 558, 700]
imag_labels = ["Very Low", "Low", "High", "Very High"]

df_filtered["imageability_bin"] = pd.cut(
    df_filtered["Imageability"], bins=imag_bins, labels=imag_labels
)

print("\nMean gate values by imageability level:")
print(df_filtered.groupby("imageability_bin")["gate"].agg(["mean", "std", "count"]))
