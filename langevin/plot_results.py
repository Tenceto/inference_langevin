import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score


def plot_results(filename, legend):
    methods = list(legend.keys())
    results = pd.read_csv(filename, index_col=0, sep=";")
    print("Samples used:", len(results) // (results.num_obs.nunique()))
    for col in ["real_values"] + [f"pred_{method}" for method in methods]:
        results[col] = results[col].apply(lambda x: np.array(eval(x)))
    for method in methods:
        results[legend[method]] = results.apply(lambda row: roc_auc_score(row["real_values"], row[f"pred_{method}"]), axis=1)
    results.groupby("num_obs")[[col for col in results.columns if col in legend.values()]].mean().plot(marker="o")
    plt.xlabel("Number of observations")
    plt.ylabel("AUCROC")
    plt.xticks(results.num_obs.unique())
    plt.grid()
    plt.show()
