import numpy as np
import torch
import pickle as pkl
import networkx as nx
from contextlib import redirect_stdout
import logging
import traceback
import sys
from inspect import signature
import pandas as pd
import os
from sklearn.metrics import f1_score

import topology_inference.initial_estimators  as init
import topology_inference.utils as ut

# np.set_printoptions(suppress=True)

# torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)

logger_file = "tuning_adam_init.log"
graph_type = "deezer_ego"
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
# obs_ratio_list = [0.15, 0.43, 0.72, 1.]
obs_ratio_list = np.logspace(np.log10(0.15), np.log10(50), 10)
# Filter parameter distribution
theta_min, theta_max = -0.5, 0.5
h_theta = ut.poly_second_order
# Prior score model
model = ut.load_model(model_file)

# Adam parameters
lr_init = 0.005
n_epochs_init = 1000
n_samples_list = [1, 5, 10, 25]
margin_list = [0.1, 0.2, 0.3, 0.4]

# Save results
output_file = f"outputs/adaminit_graphtype_{graph_type}_seed_{seed}_theta_range_{(theta_min, theta_max)}_filter_{h_theta.__name__}.csv"
theta_dist = torch.distributions.Uniform(theta_min, theta_max)
len_theta = len(signature(h_theta).parameters) - 1
n_samples_grid, margin_grid = np.meshgrid(n_samples_list, margin_list)
n_samples_grid = n_samples_grid.flatten()
margin_grid = margin_grid.flatten()

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

                num_nodes = A.shape[0]
                max_num_obs = int(np.ceil(obs_ratio_list[-1] * num_nodes))

                Y, theta = ut.simulate_data_white_noise(A, max_num_obs, theta_dist, h_theta)
                unknown_idxs = torch.triu_indices(num_nodes, num_nodes, offset=1).cuda()

                adam_init = init.AdamInitializer(h_theta=h_theta, theta_dist=theta_dist, lr=lr_init, n_epochs=n_epochs_init)
                
                for obs_ratio in obs_ratio_list:
                    num_obs = int(np.ceil(obs_ratio * num_nodes))
                    this_Y = Y[:, :num_obs]

                    np.random.seed((seed + 1) * 3)
                    torch.manual_seed((seed + 1) * 3)

                    for n_samples_init, margin in zip(n_samples_grid, margin_grid):
                        A_init_adam, _ = adam_init.initial_estimation(Y=this_Y, l1_penalty=0.0, num_runs=n_samples_init, margin=margin)

                        a_pred = A_init_adam[unknown_idxs[0], unknown_idxs[1]].cpu().numpy()
                        a_true = A[unknown_idxs[0], unknown_idxs[1]].cpu().numpy()

                        prop_unknown, prop_known_correct, prop_correct = ut.compute_initalizer_metrics(a_pred, a_true)
                        output_results.append({"obs_ratio": obs_ratio, "n_samples": n_samples_init, "margin": margin,
                                               "prop_unknown": prop_unknown, "prop_known_correct": prop_known_correct, "prop_correct": prop_correct})
                
                    logger.info(f"Finished iteration.")
                pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))
            except RuntimeError:
                logger.exception(f"Runtime error in iteration.")
                continue
    
    logger.info(f"Finished all iterations.")
