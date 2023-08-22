import numpy as np
import torch
from easydict import EasyDict as edict
from edp_gnn.utils.loading_utils import get_score_model
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def load_model(model_file):
    ckp = torch.load(model_file)
    model_config = edict(ckp['config'])
    model = get_score_model(config=model_config, dev="cuda")
    model.load_state_dict(ckp['model'], strict=False)
    model.to("cuda")
    model.eval()

    return model

def score_edp_wrapper(model, nodes, num_sigmas, max_nodes):
	node_flag_init = torch.tensor([1, 0], device="cuda")
	node_flags = node_flag_init.repeat_interleave(torch.tensor([nodes, max_nodes - nodes], device="cuda"))
	node_flags = node_flags.repeat(num_sigmas, 1)
	x = torch.zeros((num_sigmas, max_nodes, 1), device="cuda")

	def score_fun(A_tilde, sigma_idx):
		model_input = torch.zeros((num_sigmas, max_nodes, max_nodes), device="cuda")
		model_input[num_sigmas - sigma_idx - 1, :, :] = pad_adjs(A_tilde, max_nodes)

		with torch.no_grad():
			all_score_levels = model(x, model_input, node_flags)
			selected_score = all_score_levels[num_sigmas - sigma_idx - 1].detach()
		return selected_score
	
	return score_fun

def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = torch.concatenate([a, torch.zeros(ori_len, node_number - ori_len)], axis=-1)
    a = torch.concatenate([a, torch.zeros(node_number - ori_len, node_number)], axis=0)
    return a

def heat_diffusion_filter(A, theta):
    L = compute_laplacian(A)
    return torch.linalg.matrix_exp(- L * theta)
    # return - 0.2 * L - torch.eye(L.shape[0])

def arma_gf_order_one(A, alpha, beta):
    I = torch.eye(A.shape[0])
    return (I - beta * A) @ (I - alpha * A).T

def poly_third_order(A, a, b, c, d):
    # L = compute_laplacian(A)
    I = torch.eye(A.shape[0])
    return a * torch.matrix_power(A, 3) + b * torch.matrix_power(A, 2) + c * A + d * I
    # return a * (L @ L) + b * L + c * I

def poly_second_order(A, a, b, c):
    # L = compute_laplacian(A)
    I = torch.eye(A.shape[0])
    return a * torch.matrix_power(A, 2) + b * A + c * I
    # return a * (L @ L) + b * L + c * I

def compute_laplacian(A):
    D = torch.diag(A.sum(axis=1))
    return D - A

def compute_aucroc(A, A_est, use_idxs=None, return_threshold=False):
    if use_idxs is None:
        use_idxs = torch.triu_indices(A.shape[0], A.shape[1], offset=1)
    a = A[use_idxs[0], use_idxs[1]].cpu().numpy()
    a_est = A_est[use_idxs[0], use_idxs[1]].cpu().numpy()
    if return_threshold:
        fpr, tpr, thresholds = roc_curve(a, a_est)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return roc_auc_score(a, a_est), optimal_threshold
    else:
         return roc_auc_score(a, a_est)
    
def compute_f1(A, A_est, use_idxs=None):
    if use_idxs is None:
        use_idxs = torch.triu_indices(A.shape[0], A.shape[1], offset=1)
    a = A[use_idxs[0], use_idxs[1]].cpu().numpy()
    a_est = A_est[use_idxs[0], use_idxs[1]].cpu().numpy()
    return f1_score(a, a_est > 0.5)

def compute_relative_error(theta, theta_est):
    # TODO: This is a quick fix because arma_gf_order_one is symmetric w.r.t. alpha and beta
    # and the estimated coefficients could be swapped
    if len(theta) == 1:
         return (torch.abs(theta - theta_est) / theta).mean()
    elif len(theta) == 2:
        error_1 = (torch.abs(theta - theta_est) / theta).mean()
        error_2 = (torch.abs(theta - torch.flip(theta_est, dims=[0])) / theta).mean()
        return torch.min(error_1, error_2)
    else:
        raise ValueError("len(theta) > 2 is not supported yet.")

def create_partially_known_graph(A, p_unknown):
    triu_idxs = torch.triu_indices(A.shape[0], A.shape[0], offset=1)
    num_edges = len(triu_idxs[0])
    selected_unknown = torch.multinomial(torch.tensor([1 / num_edges] * num_edges), 
                                         int(num_edges * p_unknown), 
                                         replacement=False)
    A_nan = A.clone()
    unknown_idxs = triu_idxs[0][selected_unknown], triu_idxs[1][selected_unknown]
    A_nan[unknown_idxs[0], unknown_idxs[1]] = torch.nan
    A_nan[unknown_idxs[1], unknown_idxs[0]] = torch.nan
    return A_nan

def generate_grid_graph(m_min, m_max, min_nodes, max_nodes, min_random_edges, max_random_edges):
    m_values = np.arange(m_min, m_max + 1)
    possible_tuples = list(itertools.product(m_values, m_values))
    possible_tuples = [(x, y) for x, y in possible_tuples if min_nodes <= x * y <= max_nodes and x <= y]
    # print(possible_tuples)

    grid_dims = random.choices(possible_tuples, k=1)[0]
    n_random_edges = np.random.randint(min_random_edges, max_random_edges, 1)

    graph = nx.grid_2d_graph(grid_dims[0], grid_dims[1])
    A = add_random_edges(nx.to_numpy_array(graph), n_random_edges)
    return nx.from_numpy_matrix(A)


def add_random_edges(A, n_added_edges):
    if n_added_edges != 0:
        zero_indices = np.argwhere(A == 0)
        zero_indices = zero_indices[zero_indices[:, 0] != zero_indices[:, 1]]

        # Choose a random zero to convert to a one
        idx = np.random.choice(np.arange(zero_indices.shape[0]), n_added_edges, replace=False)
        row, col = zero_indices[idx, 0], zero_indices[idx, 1]

        # Convert the chosen zero to a one
        A[row, col] = 1.0
        A[col, row] = 1.0
    else:
        pass

    return A