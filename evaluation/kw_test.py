import pandas as pd
from scipy import stats
import seaborn as sns

df = pd.read_csv("./gate_soft_per_feature/word_gate_pos.csv")


groups = [group["gate"].values for name, group in df.groupby("pos")]

h_stat, p_kruskal = stats.kruskal(*groups)
print(f"Kruskal wallis H: {h_stat}, P: {p_kruskal}")
