import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score


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
    results.groupby("num_obs")[[col for col in results.columns if col in legend.values()]].mean().plot(marker="o")
    plt.xlabel("Number of observations")
    plt.ylabel("Relative error")
    plt.xticks(results.num_obs.unique())
    plt.grid()
    plt.show()

def plot_results(filename, legend, styles=None, figsize=(10, 5), output=None, x_label="num_obs",
                 thresholds=None, theta_metric=None, pad_inches=0.0, agg_fun="mean",
                 colors=None, theta_methods=None,
                 markersize=7, log_scale=False):
    if styles is not None:
        assert len(styles) == len(legend), "Number of styles must be equal to number of methods."
    else:
        styles = ["o-", "s-", "v-", "p-", "d-", "h-", "x-", "+-", "*-"]

    if theta_methods is None:
        theta_methods = legend.values()

    if theta_metric is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = [ax]
    else:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
    methods = list(legend.keys())
    results = pd.read_csv(filename, index_col=0, sep=";")
    print("Samples used:", len(results) // (results[x_label].nunique()))

    for col in ["real_graph"] + [f"graph_{method}" for method in methods]:
        results[col] = results[col].apply(lambda x: np.array(eval(x)))
    if thresholds is None:
        # AUCROC on A estimation if there are no thresholds
        for method in methods:
            results[legend[method]] = results.apply(lambda row: roc_auc_score(row["real_graph"], row[f"graph_{method}"]), axis=1)
        ax[0].set_title(r"AUCROC on $\mathbf{A}^{\mathcal{U}}$")
    else:
        # F1-score if thresholds are passed
        for method in methods:
            th = thresholds[method]
            results[legend[method]] = results.apply(lambda row: f1_score(row["real_graph"], row[f"graph_{method}"] > th), axis=1)
        ax[0].set_title(r"F1-score on $\mathbf{A}^{\mathcal{U}}$")
    if agg_fun == "mean":
        results.groupby(x_label)[[col for col in results.columns if col in legend.values()]].mean().plot(style=styles, ax=ax[0], ms=markersize, color=colors.values())
    elif agg_fun == "median":
        results.groupby(x_label)[[col for col in results.columns if col in legend.values()]].median().plot(style=styles, ax=ax[0], ms=markersize, color=colors.values())

    # Relative error on theta estimation
    if theta_metric is not None:
        try:
            for col in ["real_theta"] + [f"theta_{method}" for method in methods]:
                results[col] = results[col].astype(float)
        except ValueError:
            for col in ["real_theta"] + [f"theta_{method}" for method in methods]:
                results[col] = results[col].apply(lambda x: np.array(eval(x)))
        if theta_metric == "relative_error":
            for method in methods:
                results[legend[method]] = results.apply(lambda row: np.abs(row["real_theta"] - row[f"theta_{method}"] / row["real_theta"]).sum(), axis=1)
            ax[1].set_title(r"Relative error on $\pmb{\theta}$")
        elif theta_metric == "mse":
            for method in methods:
                results[legend[method]] = results.apply(lambda row: np.sqrt(np.sum((row["real_theta"] - row[f"theta_{method}"]) ** 2) / np.sum(row["real_theta"] ** 2)), axis=1)
            ax[1].set_title(r"Normalized RMSE on $\pmb{\theta}$")
        if agg_fun == "mean":
            results.groupby(x_label)[[col for col in results.columns if col in theta_methods]].mean().plot(style=styles, ax=ax[1], ms=markersize, color=colors.values())
        elif agg_fun == "median":
            results.groupby(x_label)[[col for col in results.columns if col in theta_methods]].median().plot(style=styles, ax=ax[1], ms=markersize, color=colors.values())
        # Remove the legend from both plots
        for a in ax:
            a.get_legend().remove()
    
    if theta_metric is not None:
        # Add a unique legend to the side of both plots
        ax[1].legend(legend.values(), loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax[0].legend(legend.values(), loc='center left', bbox_to_anchor=(1, 0.5))

    for a in ax:
        if x_label == "num_obs":
            a.set_xlabel(r"$K$")
        elif x_label == "obs_ratio":
            a.set_xlabel(r"$K/N$")
        if log_scale:
            a.set_xscale("log")
        a.get_xaxis().set_major_formatter(tick.ScalarFormatter())
        a.set_xticks(results[x_label].unique(), labels=results[x_label].unique())
        a.set_xticks([], minor=True)
        # Set only horizontal grid
        a.grid(axis="y")

    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, pad_inches=pad_inches, bbox_inches='tight')


def compute_metrics(filename, legend, thresholds=None, theta_metric=None):
    methods = list(legend.keys())
    results = pd.read_csv(filename, index_col=0, sep=";")
    for col in [col for col in results.columns if "theta" in col]:
        results[col] = results[col].str.replace("nan", "np.nan")
    for col in ["real_graph"] + [f"graph_{method}" for method in methods]:
        results[col] = results[col].apply(lambda x: np.array(eval(x)))
    if thresholds is None:
        # AUCROC on A estimation if there are no thresholds
        for method in methods:
            results[legend[method] + " Graph"] = results.apply(lambda row: roc_auc_score(row["real_graph"], row[f"graph_{method}"]), axis=1)
    else:
        # F1-score if thresholds are passed
        for method in methods:
            th = thresholds[method]
            results[legend[method] + " Graph"] = results.apply(lambda row: f1_score(row["real_graph"], row[f"graph_{method}"] > th), axis=1)

    if theta_metric is not None:
        try:
            for col in ["real_theta"] + [f"theta_{method}" for method in methods]:
                results[col] = results[col].astype(float)
        except ValueError:
            for col in ["real_theta"] + [f"theta_{method}" for method in methods]:
                results[col] = results[col].apply(lambda x: np.array(eval(x)))
        if theta_metric == "relative_error":
            for method in methods:
                results[legend[method] + " Theta"] = results.apply(lambda row: np.abs(row["real_theta"] - row[f"theta_{method}"] / row["real_theta"]).sum(), axis=1)
        elif theta_metric == "mse":
            for method in methods:
                results[legend[method] + " Theta"] = results.apply(lambda row: np.sqrt(np.sum((row["real_theta"] - row[f"theta_{method}"]) ** 2) / np.sum(row["real_theta"] ** 2)), axis=1)

    results = results[["obs_ratio"] + [col for col in results.columns if "Graph" in col or "Theta" in col]]

    return results


def plot_l1_tuning(filename, graph_metric="aucroc", figsize=(10, 5)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    df = pd.read_csv(filename, sep=";", index_col=0)
    print("Samples used:", len(df) // (df.num_obs.nunique() * df.l1_penalty.nunique()))
    df = df.rename(columns={"num_obs": "Number of observations"})
    df = df.groupby(["Number of observations", "l1_penalty"])[[graph_metric, "rel_error"]].mean()

    df[graph_metric].unstack(level=0).plot(ax=ax[0])
    ax[0].set_xlabel("L1 penalty")
    if graph_metric == "aucroc":
        ax[0].set_title(r"AUCROC on $\mathbf{A}$")
    elif graph_metric == "f1":
        ax[0].set_title(r"F1 score on $\mathbf{A}$")
    ax[0].set_xscale("log")
    ax[0].grid()

    df["rel_error"].unstack(level=0).plot(ax=ax[1])
    ax[1].set_xlabel("L1 penalty")
    ax[1].set_title(r"Relative error on $\theta$")
    ax[1].set_xscale("log")
    ax[1].grid()
    plt.tight_layout()
    plt.show()
