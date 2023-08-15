import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score


def plot_graph_results(filename, legend):
    methods = list(legend.keys())
    results = pd.read_csv(filename, index_col=0, sep=";")
    print("Samples used:", len(results) // (results.num_obs.nunique()))
    for col in ["real_graph"] + [f"graph_{method}" for method in methods]:
        results[col] = results[col].apply(lambda x: np.array(eval(x)))
    for method in methods:
        results[legend[method]] = results.apply(lambda row: roc_auc_score(row["real_graph"], row[f"graph_{method}"]), axis=1)
    results.groupby("num_obs")[[col for col in results.columns if col in legend.values()]].median().plot(marker="o")
    plt.xlabel("Number of observations")
    plt.ylabel("AUCROC")
    plt.xticks(results.num_obs.unique())
    plt.grid()
    plt.show()

def plot_theta_results(filename, legend):
    methods = list(legend.keys())
    results = pd.read_csv(filename, index_col=0, sep=";")
    print("Samples used:", len(results) // (results.num_obs.nunique()))
    for col in ["real_theta"] + [f"theta_{method}" for method in methods]:
        results[col] = results[col].astype(float)
    for method in methods:
        results[legend[method]] = results.apply(lambda row: np.abs(row["real_theta"] - row[f"theta_{method}"]) / row["real_theta"], axis=1)
    results.groupby("num_obs")[[col for col in results.columns if col in legend.values()]].median().plot(marker="o")
    plt.xlabel("Number of observations")
    plt.ylabel("Relative error")
    plt.xticks(results.num_obs.unique())
    plt.grid()
    plt.show()

def plot_results(filename, legend, figsize=(10, 5), output=None):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    methods = list(legend.keys())
    results = pd.read_csv(filename, index_col=0, sep=";")
    print("Samples used:", len(results) // (results.num_obs.nunique()))

    # AUCROC on A estimation
    for col in ["real_graph"] + [f"graph_{method}" for method in methods]:
        results[col] = results[col].apply(lambda x: np.array(eval(x)))
    for method in methods:
        results[legend[method]] = results.apply(lambda row: roc_auc_score(row["real_graph"], row[f"graph_{method}"]), axis=1)
    results.groupby("num_obs")[[col for col in results.columns if col in legend.values()]].mean().plot(marker="o", ax=ax[0])
    ax[0].set_title(r"AUCROC on $\mathbf{A}$")
    ax[0].set_xlabel("Number of observations")
    ax[0].set_xticks(results.num_obs.unique())
    ax[0].grid()

    # Relative error on theta estimation
    for col in ["real_theta"] + [f"theta_{method}" for method in methods]:
        results[col] = results[col].astype(float)
    for method in methods:
        results[legend[method]] = results.apply(lambda row: np.abs(row["real_theta"] - row[f"theta_{method}"]) / row["real_theta"], axis=1)
    results.groupby("num_obs")[[col for col in results.columns if col in legend.values()]].mean().plot(marker="o", ax=ax[1])
    ax[1].set_title(r"Relative error on $\theta$")
    ax[1].set_xlabel("Number of observations")
    ax[1].set_xticks(results.num_obs.unique())
    ax[1].grid()

    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output)

