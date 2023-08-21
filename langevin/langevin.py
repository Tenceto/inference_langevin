import torch
from inspect import signature

from langevin.utils import compute_aucroc, compute_relative_error


class LangevinEstimator:
    def __init__(self, h_theta, A_score_model, sigma_e, theta_prior_dist):
        self.h_theta = h_theta
        self.num_filter_params = len(signature(h_theta).parameters) - 1
        self.A_score_model = A_score_model
        self.sigma_e = sigma_e
        self.theta_prior_dist = theta_prior_dist
        self.metrics = None

    def score_graph_likelihood(self, A, X, Y, theta):
        A = A.clone().requires_grad_(True)
        # We only need the upper triangular part of A to account
        # for the symmetry of the matrix, so that the gradient is correct
        log_likelihood = - self.compute_minus_likelihood(torch.triu(A) + torch.triu(A, diagonal=1).T, 
                                                         X, Y, theta)
        grad = torch.autograd.grad(log_likelihood, A)[0]
        return grad

    def compute_minus_likelihood(self, A, X, Y, theta):
        F = self.h_theta(A, *theta)
        log_likelihood = - 1 / (2 * self.sigma_e ** 2) * (torch.linalg.norm(Y - F @ X, dim=0) ** 2).sum()
        return - log_likelihood

    def langevin_estimate(self, A_nan, X, Y, 
                          sigmas_sq, epsilon, temperature, steps, adam_lr,
                          projection_method="rounding", clip_A_tilde=False, true_A=None, true_theta=None):
        
        # Initialize theta_tilde
        theta_tilde = self.theta_prior_dist.sample([self.num_filter_params]).abs()
        # We define this tensor to be used for the optimizer without screwing up the gradients
        theta_grad = theta_tilde.requires_grad_(True)
        optimizer = torch.optim.Adam([theta_grad], lr=adam_lr)

        unknown_idxs = torch.where(torch.isnan(torch.triu(A_nan)))
        known_mask = ~ torch.isnan(A_nan)
        size_unknown = len(unknown_idxs[0])
        # Distribution for the Langevin noise (NOT the annealing one, this one has unitary variance)
        z_dist = torch.distributions.MultivariateNormal(torch.zeros(size_unknown), torch.eye(size_unknown))

        # Initialize A_tilde and its projection
        A_tilde = torch.distributions.Normal(0.5, 0.1).sample(A_nan.shape)
        A_tilde = 0.5 * (torch.triu(A_tilde) + torch.triu(A_tilde, 1).T)
        A_tilde.fill_diagonal_(0.0)
        A_tilde[known_mask] = A_nan[known_mask]
        A_proj = self.project_adjacency_matrix(A_tilde, projection_method)

        if true_A is not None and true_theta is not None:
            compute_metrics = True
            metrics = {"aucroc": torch.empty(steps * len(sigmas_sq)),
                       "relative_error": torch.empty(steps * len(sigmas_sq))}

        for sigma_i_idx, sigma_i_sq in enumerate(sigmas_sq):
            alpha = epsilon * sigma_i_sq / sigmas_sq[-1]
            for t in range(steps):
                z = z_dist.sample([1])
                # Compute the score
                score_prior_A = self.A_score_model(A_tilde, sigma_i_idx)[unknown_idxs[0], unknown_idxs[1]]
                score_likelihood_A = self.score_graph_likelihood(A_tilde, X, Y, theta_tilde)[unknown_idxs[0], unknown_idxs[1]]
                # Update A
                A_tilde[unknown_idxs[0], unknown_idxs[1]] = (A_tilde[unknown_idxs[0], unknown_idxs[1]]
                                                             + alpha * (score_prior_A + score_likelihood_A)
                                                             + torch.sqrt(2 * alpha * temperature) * z)
                A_tilde[unknown_idxs[1], unknown_idxs[0]] = A_tilde[unknown_idxs[0], unknown_idxs[1]]
                if clip_A_tilde:
                    A_tilde = self.clip_adjacency_matrix(A_tilde, torch.sqrt(sigma_i_sq))
                A_proj = self.project_adjacency_matrix(A_tilde, projection_method)

                # Update theta
                optimizer.zero_grad()
                loss = self.compute_minus_likelihood(A_proj, X, Y, theta_grad)
                loss.backward()
                optimizer.step()
                # TODO: This is only for the exponential filter, because if theta is negative the filter will diverge
                with torch.no_grad():
                    theta_tilde = torch.clip(theta_grad.detach().clone(), min=0.0)
                    # Compute metrics
                    if compute_metrics:
                        n_step = steps * sigma_i_idx + t
                        metrics["aucroc"][n_step] = compute_aucroc(true_A, A_tilde, use_idxs=unknown_idxs)
                        metrics["relative_error"][n_step] = compute_relative_error(true_theta, theta_tilde)

        return A_tilde, theta_tilde, metrics
    
    def project_adjacency_matrix(self, A_tilde, projection_method):
        if projection_method == "rounding":
            idx_1 = torch.where(A_tilde >= 0.5)
            A_proj = torch.zeros_like(A_tilde)
            A_proj[idx_1] = 1.0
            return A_proj
        elif projection_method == "random":
            A_proj = torch.randint(0, 2, A_tilde.shape).double()
            A_proj = torch.triu(A_proj) + torch.triu(A_proj, 1).T
            A_proj.fill_diagonal_(0.0)
            return A_proj
        elif projection_method == "copy":
            return A_tilde.clone()
    
    def clip_adjacency_matrix(self, A_tilde, sigma_i):
        min_clip, max_clip = 0.0 - sigma_i, 1.0 + sigma_i
        return torch.clip(A_tilde, min_clip, max_clip)


class AdamEstimator:
    def __init__(self, h_theta, sigma_e, lr, n_iter):
        self.h_theta = h_theta
        self.sigma_e = sigma_e
        self.lr = lr
        self.n_iter = n_iter
        self.num_filter_params = len(signature(h_theta).parameters) - 1

    def adam_estimate(self, A_nan, X, Y, theta_prior_dist, l1_penalty):
        A_tilde = torch.distributions.Normal(0.5, 0.1).sample(A_nan.shape)
        A_tilde = 0.5 * (torch.triu(A_tilde) + torch.triu(A_tilde, 1).T)
        A_tilde.fill_diagonal_(0.0)
        
        unknown_mask = torch.isnan(A_nan)
        A_tilde[~ unknown_mask] = A_nan[~ unknown_mask]

        A_known = A_tilde.clone()
        unknown_mask = unknown_mask.float()

        A_tilde.requires_grad_(True)
        theta = theta_prior_dist.sample([self.num_filter_params]).abs()
        theta.requires_grad_(True)
        optimizer = torch.optim.Adam([A_tilde, theta], lr=self.lr)
        loss_hist = []

        for _ in range(self.n_iter):
            A = self.symmetrize_and_clip(A_tilde, A_nan, unknown_mask)
            optimizer.zero_grad()
            loss = self.compute_penalized_minus_likelihood(A, X, Y, theta)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

        A = self.symmetrize_and_clip(A_tilde.detach(), A_nan, unknown_mask)
        
        return A, theta, loss_hist
    
    def compute_penalized_minus_likelihood(self, A, X, Y, theta):
        A_symmetric = torch.triu(A) + torch.triu(A, diagonal=1).T
        F = self.h_theta(A_symmetric, *theta)
        log_likelihood = - 1 / (2 * self.sigma_e ** 2) * (torch.linalg.norm(Y - F @ X, dim=0) ** 2).sum()
        return - log_likelihood + self.l1_penalty * A_symmetric.abs().sum()
    
    def symmetrize_and_clip(self, A, A_nan, unknown_mask):
        A = torch.triu(A) + torch.triu(A, diagonal=1).T
        A = torch.clip(A, 0.0, 1.0)
        A[~ unknown_mask] = A_nan[~ unknown_mask]
        return A
