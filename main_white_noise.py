import numpy as np
import torch
import pickle as pkl
import networkx as nx
from contextlib import redirect_stdout
import logging
import traceback
import sys
import pandas as pd
import os
from functools import partial

import topology_inference.estimators_white_noise  as est
import topology_inference.initial_estimators as init
import topology_inference.utils as ut

# np.set_printoptions(suppress=True)

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)

logger_file = "full_benchmark.log"
graph_type = "grids"
seed = 130
n_graphs_test = 100

torch.manual_seed(seed)
np.random.seed(seed)

model_files = {
    "deezer_ego": ("edp_gnn/exp/deezer_ego/edp-gnn_train_deezer_ego__Jun-14-14-14-11_1489048/models/" +
                   "train_deezer_ego_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth"),
    "barabasi": ("edp_gnn/exp/barabasi_albert_diff_nodes/edp-gnn_barabasi_albert_[47, 49, 51, 53]__Jun-13-10-13-20_999031/models/" + 
                 "barabasi_albert_[47, 49, 51, 53]_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth"),
    "barabasi_smaller": ("edp_gnn/exp/barabasi_albert_diff_nodes_small/edp-gnn_barabasi_albert_[15, 17, 19, 21, 23]__Aug-07-15-50-45_2986529/models/" + 
                         "barabasi_albert_[15, 17, 19, 21, 23]_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth"),
     "grids": ("edp_gnn/exp/grids_dif_nodes/edp-gnn_grids_dif_nodes__Feb-12-22-48-21_4158697/models/" +
              "grids_dif_nodes_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")

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
    graphs = [nx.dual_barabasi_albert_graph(np.random.randint(47, 53), 2, 4, 0.5, seed=seed) for _ in range(n_graphs_test)]
    max_nodes = 53
elif graph_type == "barabasi_smaller":
    graphs = [nx.dual_barabasi_albert_graph(np.random.randint(15, 23), 2, 4, 0.5, seed=seed) for _ in range(n_graphs_test)]
    max_nodes = 23
elif graph_type == "grids":
    m_min = 5
    m_max = 9
    min_nodes = 40
    max_nodes = 49
    min_random_edges = 2
    max_random_edges = 5
    seed = 0
    graphs = [ut.generate_grid_graph(m_min, m_max, min_nodes, max_nodes, min_random_edges, max_random_edges) for _ in range(n_graphs_test)]
    max_nodes = 49

model_file = model_files[graph_type]
adj_matrices = [torch.tensor(nx.to_numpy_array(g, nodelist=np.random.permutation(g.nodes()))).cuda() for g in graphs]
# Number of measurements
obs_ratio_list = [1.0, 1.5, 2.0, 2.5, 3.0]
# Filter parameter distribution
theta_min, theta_max = 1.0, 2.0
# order_p, order_q = 2, 0
# h_theta = partial(ut.rational, order_p=order_p, order_q=order_q)
# h_theta.__name__ = f"rational_{order_p}_{order_q}"
# len_theta_fun = lambda A: order_p + 1 + order_q
h_theta = ut.node_varying_second_order
len_theta_fun = lambda A: 3 * A.shape[0]
# Prior score model
model = ut.load_model(model_file)

# Langevin parameters
sigmas = torch.linspace(0.5, 0.03, 10)
epsilon = 1.0E-6
steps = 300
num_samples = 1
temperature = 0.5

# Adam parameters
lr = 0.01
n_epochs = 500
l1_penalty = 0.0

# Initializer parameters
lr_init = 0.01
n_epochs_init = 1000
n_samples_init = 20
margin = 0.3

# Spectral template parameters
# threshold_fun = hyperparams.select_spectral_threshold(graph_type, (theta_min, theta_max), h_theta)
threshold_fun = lambda ratio: 0.01832276 * np.log(ratio) + 0.11801762

# GLasso parameters
n_bootstrap = 50

# Save results
output_file = f"outputs/v2_fullbenchmark_graphtype_{graph_type}_seed_{seed}_temperature_{temperature}_theta_range_{(theta_min, theta_max)}_filter_{h_theta.__name__}.csv"
theta_dist = torch.distributions.Uniform(theta_min, theta_max)

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
    def flush():
        pass
    setattr(logger, 'flush', flush)

    # Install exception handler
    sys.excepthook = exc_handler
    
    with redirect_stdout(logger):
        for seed, A in enumerate(adj_matrices):
            try:
                output_results = list()
                np.random.seed(seed)
                torch.manual_seed(seed)

                num_nodes = A.shape[0]
                max_num_obs = int(np.ceil(obs_ratio_list[-1] * num_nodes))

                len_theta = len_theta_fun(A)

                A_score_model = ut.score_edp_wrapper(model, A.shape[0], len(sigmas), max_nodes=max_nodes)
                # A_score_zero = lambda A, _: torch.zeros(A.shape)

                Y, theta = ut.simulate_data_white_noise(A, max_num_obs, theta_dist, h_theta, len_theta)
                unknown_idxs = torch.triu_indices(num_nodes, num_nodes, offset=1).cuda()

                # Initializer
                boot_init = init.BootstrapAdamInitializer(h_theta=h_theta, 
                                                          len_theta=len_theta,
                                                          theta_dist=theta_dist, 
                                                          lr=lr_init, 
                                                          n_epochs=n_epochs_init)
                # Estimators
                langevin_posterior_est = est.LangevinEstimator(h_theta=h_theta,
                                                               len_theta=len_theta,
                                                               A_score_model=A_score_model,
                                                               theta_prior_dist=theta_dist)
                adam_est = est.AdamEstimator(h_theta=h_theta,
                                             len_theta=len_theta,
                                             theta_prior_dist=theta_dist, 
                                             lr=lr, 
                                             n_iter=n_epochs)
                spectral_est = est.SpectralTemplates(h_theta, len_theta,
                                                     threshold_fun, 
                                                     epsilon_range=(0,2), 
                                                     num_iter_reweight_refinements=3)
                glasso_est = est.StabilitySelector(n_bootstrap=n_bootstrap,
                                                   h_theta=h_theta,
                                                   len_theta=len_theta,
                                                   n_jobs=10)

                for obs_ratio in obs_ratio_list:
                    num_obs = int(np.ceil(obs_ratio * num_nodes))
                    this_Y = Y[:, :num_obs]

                    np.random.seed((seed + 1) * 3)
                    torch.manual_seed((seed + 1) * 3)

                    # Adam initializations
                    A_init_boot, _ = boot_init.initial_estimation(Y=this_Y, l1_penalty=0.0, 
                                                                  bootstrap_samples=n_samples_init, 
                                                                  margin=margin)
                    # Langevin estimations
                    A_lang_boot, theta_lang_boot = langevin_posterior_est.langevin_estimate(A_nan=A_init_boot, Y=this_Y, 
                                                                                            sigmas_sq=sigmas ** 2, epsilon=epsilon, 
                                                                                            temperature=temperature, steps=steps,
                                                                                            adam_lr=lr, num_samples=num_samples,
                                                                                            projection_method="copy", clip_A_tilde=True)
                    # Adam estimations
                    A_adam_boot, theta_adam_boot, _ = adam_est.adam_estimate(A_nan=A_init_boot, Y=this_Y, l1_penalty=0.0)

                    # Spectral template estimations
                    A_spectral, theta_spectral = spectral_est.spectral_estimate(Y=this_Y, estimate_theta=False)

                    # GLasso estimations
                    A_glasso, theta_glasso = glasso_est.glasso_estimate(Y=this_Y, estimate_theta=False)

                    A_all = {
                        "langevin_boot": A_lang_boot,
                        "adam_boot": A_adam_boot,
                        "spectral": A_spectral,
                        "glasso": A_glasso,
                    }
                    # theta_all = {
                    #     "langevin_boot": theta_lang_boot,
                    #     "adam_boot": theta_adam_boot,
                    #     "spectral": theta_spectral,
                    #     "glasso": theta_glasso,
                    # }

                    this_output = dict()

                    this_output["obs_ratio"] = obs_ratio
                    # this_output["num_samples"] = num_samples
                    this_output["real_graph"] = A[unknown_idxs[0], unknown_idxs[1]].cpu().numpy().tolist()
                    # if len_theta > 1:
                    #     this_output["real_theta"] = theta.cpu().numpy().tolist()
                    # else:
                    #     this_output["real_theta"] = theta.cpu().item()
                    for method, A_est in A_all.items():
                        this_output[f"graph_{method}"] = A_est[unknown_idxs[0], unknown_idxs[1]].cpu().numpy().tolist()
                        # if len_theta > 1:
                        #     this_output[f"theta_{method}"] = theta_all[method].cpu().numpy().tolist()
                        # else:
                        #     this_output[f"theta_{method}"] = theta_all[method].cpu().item()
                    output_results.append(this_output)

                    ratio_unknown = torch.isnan(A_init_boot[unknown_idxs[0], unknown_idxs[1]]).float().mean().item()
                    logger.info(f"Finished iteration. Seed: {seed}, k/n = {obs_ratio}, k = {num_obs}, n = {num_nodes}, |U| / dim(a): {ratio_unknown}")
                pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))
            except RuntimeError:
                logger.exception(f"Runtime error in iteration.")
                continue
    
    logger.info(f"Finished all iterations.")
