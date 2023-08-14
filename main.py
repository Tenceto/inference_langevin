import numpy as np
import torch
import pickle as pkl
import networkx as nx
from contextlib import redirect_stdout
import logging
import traceback
import sys
from functools import partial
import pandas as pd
import os

from langevin import langevin as lang
import langevin.utils as ut

# np.set_printoptions(suppress=True)

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)

logger_file = "langevin.log"
graph_type = "deezer_ego"
seed = 13
n_graphs_test = 100

torch.manual_seed(seed)
np.random.seed(seed)

model_files = {
    "deezer_ego": ("edp_gnn/exp/deezer_ego/edp-gnn_train_deezer_ego__Jun-14-14-14-11_1489048/models/" +
                   "train_deezer_ego_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth"),
    "barabasi": ("edp_gnn/exp/barabasi_albert_diff_nodes/edp-gnn_barabasi_albert_[47, 49, 51, 53]__Jun-13-10-13-20_999031/models/" + 
                 "barabasi_albert_[47, 49, 51, 53]_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth"),
    "barabasi_smaller": ("edp_gnn/exp/barabasi_albert_diff_nodes_small/edp-gnn_barabasi_albert_[15, 17, 19, 21, 23]__Aug-07-15-50-45_2986529/models/" + 
                         "barabasi_albert_[15, 17, 19, 21, 23]_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
}

if graph_type == "deezer_ego":
    with open("edp_gnn/data/test_deezer_ego.pkl", "rb") as f:
        graphs = pkl.load(f)
        graphs = np.array(graphs, dtype=object)
        if n_graphs_test <= len(graphs):
            graphs = graphs[np.random.choice(np.arange(len(graphs)).astype(int), size=n_graphs_test, replace=False)]
        else:
            raise RuntimeError("Not enough graphs in the dataset.")
    max_nodes = 25
elif graph_type == "barabasi":
    graphs = [nx.dual_barabasi_albert_graph(50, 2, 4, 0.5, seed=seed) for _ in range(n_graphs_test)]
    max_nodes = 53
elif graph_type == "barabasi_smaller":
    graphs = [nx.dual_barabasi_albert_graph(50, 2, 4, 0.5, seed=seed) for _ in range(n_graphs_test)]
    max_nodes = 23

model_file = model_files[graph_type]
adj_matrices = [torch.tensor(nx.to_numpy_array(g, nodelist=np.random.permutation(g.nodes()))) for g in graphs]
# Number of measurements
n_list = [1, 5, 10, 15]
# Filter parameter distribution
theta_dist = torch.distributions.Normal(5, 5)
# Variance of the noise
sigma_e = 1
# Known fraction of the matrix
p_unknown = 0.5
# Prior score model
model = ut.load_model(model_file)

# Langevin parameters
sigmas = torch.linspace(0.5, 0.03, 10)
epsilon = 1.0E-6
steps = 300
# num_samples = 1
temperature = 0.5

# Adam parameters
lr = 0.01
n_epochs = 1000

# Save results
output_file = f"outputs/{graph_type}_{seed}_{lr}_{sigma_e}_{p_unknown}_{temperature}_{n_list}.csv"

def simulate_data(A, n):
    # Filter parameter
    theta = theta_dist.sample().abs()
    # Number of nodes
    p = A.shape[0]
    # Dynamics matrix
    F = ut.heat_diffusion_filter(A, theta)
    # L = ut.compute_laplacian(A)
    e_dist = torch.distributions.Normal(0, sigma_e)

    # Generate state and measurement sequence
    X = torch.empty((p, n)).uniform_(-10, 10) # * torch.empty((p, n)).bernoulli_(0.8)
    Y = F @ X + e_dist.sample((p, n))

    # Generate unknown adjacency matrix
    A_nan = ut.create_partially_known_graph(A, p_unknown)

    return X, Y, A_nan, theta

# Run your main script here:
if __name__ == '__main__':
    logging.basicConfig(filename=f"logs/{logger_file}",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger('Langevin')
    logger.info("Running Langevin Simulation")
    logging.getLogger('rpy2.rinterface_lib.callbacks').disabled = True

    def exc_handler(exctype, value, tb):
        logger.exception(''.join(traceback.format_exception(exctype, value, tb)))

    # Install exception handler
    sys.excepthook = exc_handler
    
    with redirect_stdout(logging):
        for seed, A in enumerate(adj_matrices):
            try:
                output_results = list()
                np.random.seed(seed)
                torch.manual_seed(seed)

                A_score_model = ut.score_edp_wrapper(model, A.shape[0], len(sigmas), max_nodes=max_nodes)
                A_score_zero = lambda A, _: torch.zeros(A.shape)

                X, Y, A_nan, theta = simulate_data(A, n_list[-1])
                known_idxs = torch.where(~ torch.isnan(A_nan))
                unknown_idxs = torch.where(torch.isnan(A_nan))

                langevin_posterior_est = lang.LangevinEstimator(h_theta=ut.heat_diffusion_filter,
                                                                A_score_model=A_score_model,
                                                                theta_prior_dist=theta_dist,
                                                                sigma_e=sigma_e)
                langevin_likelihood_est = lang.LangevinEstimator(h_theta=ut.heat_diffusion_filter,
                                                                A_score_model=A_score_zero,
                                                                theta_prior_dist=theta_dist,
                                                                sigma_e=sigma_e)
                adam_est = lang.AdamEstimator(h_theta=ut.heat_diffusion_filter,
                                              sigma_e=sigma_e, lr=lr, n_iter=n_epochs)
                
                for n in n_list:
                    this_X = X[:, :n]
                    this_Y = Y[:, :n]

                    np.random.seed((seed + 1) * 3)

                    A_adam, theta_adam, _ = adam_est.adam_estimate(A_nan=A_nan, X=this_X, Y=this_Y, theta_prior_dist=theta_dist)
                    A_posterior, theta_posterior, _ = langevin_posterior_est.langevin_estimate(A_nan=A_nan, X=this_X, Y=this_Y, 
                                                             sigmas_sq=sigmas ** 2, epsilon=epsilon, 
                                                             temperature=temperature, steps=steps,
                                                             adam_lr=lr,
                                                             projection_method="copy", clip_A_tilde=True, 
                                                             true_A=A, true_theta=theta)
                    A_likelihood, theta_likelihood, _ = langevin_likelihood_est.langevin_estimate(A_nan=A_nan, X=this_X, Y=this_Y, 
                                                             sigmas_sq=sigmas ** 2, epsilon=epsilon, 
                                                             temperature=temperature, steps=steps,
                                                             adam_lr=lr,
                                                             projection_method="copy", clip_A_tilde=True, 
                                                             true_A=A, true_theta=theta)
                    
                    A_all = {
                        "langevin_posterior": A_posterior, 
                        "langevin_likelihood": A_likelihood,
                        "adam": A_adam,
                    }
                    theta_all = {
                        "langevin_posterior": theta_posterior,
                        "langevin_likelihood": theta_likelihood,
                        "adam": theta_adam,
                    }

                    this_output = dict()

                    this_output["num_obs"] = n
                    # this_output["num_samples"] = num_samples
                    this_output["real_graph"] = A[unknown_idxs[0], unknown_idxs[1]].cpu().numpy().tolist()
                    this_output["real_theta"] = theta.cpu().item()
                    for method, A_est in A_all.items():
                        this_output[f"graph_{method}"] = A_est[unknown_idxs[0], unknown_idxs[1]].cpu().numpy().tolist()
                        this_output[f"theta_{method}"] = theta_all[method].cpu().item()
                    output_results.append(this_output)
                
                    logger.info(f"Finished iteration.")
                pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))
            except RuntimeError:
                logger.exception(f"Runtime error in iteration.")
                continue
    
    logger.info(f"Finished all iterations.")
