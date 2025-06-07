import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, kruskal, f_oneway
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./gate_soft_per_token/word_gate_concreteness.csv")

df_filtered = df[df["Concreteness"] > 0]

corr, p_corr = spearmanr(df_filtered["Concreteness"], df_filtered["gate"])
print(f"Spearman correlation: {corr:.3f}, P-value: {p_corr:.3f}")

conc_bins = [150, 318, 438, 558, 700]
conc_labels = ["Very Abstract", "Abstract", "Concrete", "Very Concrete"]


df_filtered["concreteness_bin"] = pd.cut(
    df_filtered["Concreteness"], bins=conc_bins, labels=conc_labels
)

print("\nMean gate values by concreteness level:")
print(df_filtered.groupby("concreteness_bin")["gate"].agg(["mean", "std", "count"]))
