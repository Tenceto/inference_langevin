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

def compute_laplacian(A):
    D = torch.diag(A.sum(axis=1))
    return D - A

def compute_aucroc(A, A_est, return_threshold=False):
    idxs_triu = torch.triu_indices(A.shape[0], A.shape[1], offset=1)
    a = A[idxs_triu[0], idxs_triu[1]].cpu().numpy()
    a_est = A_est[idxs_triu[0], idxs_triu[1]].cpu().numpy()
    if return_threshold:
        fpr, tpr, thresholds = roc_curve(a, a_est)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return roc_auc_score(a, a_est), optimal_threshold
    else:
         return roc_auc_score(a, a_est)