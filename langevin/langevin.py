import torch
from langevin.utils import compute_aucroc


class LangevinEstimator:
    def __init__(self, h_theta, A_score_model, sigma_e, theta_prior_dist=None, theta_fixed=None):
        self.h_theta = h_theta
        self.theta_prior_dist = theta_prior_dist
        self.A_score_model = A_score_model
        self.sigma_e = sigma_e

        if theta_fixed is not None:
            self.theta_tilde = theta_fixed
            self.estimate_theta = False
        else:
            self.theta_tilde = self.theta_prior_dist.sample()
            self.estimate_theta = True

        self.metrics = None

    def score_joint_likelihood(self, X, Y, grad_variable):
        if grad_variable == "A":
            A = self.A_tilde.clone().requires_grad_(True)
            # We only need the upper triangular part of A to account
            # for the symmetry of the matrix, so that the gradient is correct
            F = self.h_theta(torch.triu(A) + torch.triu(A, diagonal=1).T, self.theta_tilde)
        elif grad_variable == "theta":
            theta = self.theta_tilde.clone().requires_grad_(True)
            F = self.h_theta(self.A_tilde, theta)

        log_likelihood = - 1 / (2 * self.sigma_e ** 2) * (torch.linalg.norm(Y - F @ X, dim=0) ** 2).sum()
        
        if grad_variable == "A":
            grad = torch.autograd.grad(log_likelihood, A)[0]
        elif grad_variable == "theta":
            grad = torch.autograd.grad(log_likelihood, theta)[0]
            
        return grad
    
    def compute_theta_prior_score(self):
        theta = self.theta_tilde.clone().requires_grad_(True)
        log_prob = self.theta_prior_dist.log_prob(theta)
        grad = torch.autograd.grad(log_prob, theta)[0]

        return grad

    def langevin_estimate(self, X, Y, sigmas_sq, epsilon, steps, temperature, projection_method="rounding", clip_A_tilde=False, true_A=None):
        idxs_triu = torch.triu_indices(X.shape[0], X.shape[0], offset=1)
        dim_A = len(idxs_triu[0])
        dim_theta = len([self.theta_tilde])
        z_dist = torch.distributions.MultivariateNormal(torch.zeros(dim_A), torch.eye(dim_A))
        v_dist = torch.distributions.MultivariateNormal(torch.zeros(dim_theta), torch.eye(dim_theta))

        self.A_tilde = torch.distributions.Normal(0.5, 0.1).sample((X.shape[0], X.shape[0]))
        self.A_tilde = 0.5 * (torch.triu(self.A_tilde) + torch.triu(self.A_tilde, 1).T)
        self.A_tilde.fill_diagonal_(0.0)
        self.A_proj = self.project_adjacency_matrix(projection_method)

        if true_A is not None:
            compute_metrics = True
            self.metrics = {"aucroc": torch.empty(steps * len(sigmas_sq))}
        else:
            compute_metrics = False

        for sigma_i_idx, sigma_i_sq in enumerate(sigmas_sq):
            alpha = epsilon * sigma_i_sq / sigmas_sq[-1]
            # sigma_i = torch.sqrt(sigma_i_sq)
            for t in range(steps):
                if torch.any(torch.isnan(self.A_tilde)):
                    print("A_tilde contains NaNs")
                    break
                z = z_dist.sample([1])
                v = v_dist.sample([1])

                score_prior_A = self.A_score_model(self.A_tilde, sigma_i_idx)[idxs_triu[0], idxs_triu[1]]
                score_likelihood_A = self.score_joint_likelihood(X, Y, grad_variable="A")[idxs_triu[0], idxs_triu[1]]
                
                self.A_tilde[idxs_triu[0], idxs_triu[1]] = (self.A_tilde[idxs_triu[0], idxs_triu[1]]
                                                            + alpha * (score_prior_A + score_likelihood_A)
                                                            + torch.sqrt(2 * alpha * temperature) * z)
                self.A_tilde[idxs_triu[1], idxs_triu[0]] = self.A_tilde[idxs_triu[0], idxs_triu[1]]
                
                if clip_A_tilde:
                    self.A_tilde = self.clip_adjacency_matrix(torch.sqrt(sigma_i_sq))

                # Convert the current A_tilde to only 0 and 1
                self.A_proj = self.project_adjacency_matrix(projection_method)
                if self.estimate_theta:
                    score_prior_theta = self.compute_theta_prior_score()
                    score_likelihood_theta = self.score_joint_likelihood(X, Y, grad_variable="theta")
                    self.theta_tilde = self.theta_tilde + alpha * (score_prior_theta + score_likelihood_theta) + torch.sqrt(2 * alpha * temperature) * v

                if compute_metrics:
                    self.metrics["aucroc"][steps * sigma_i_idx + t] = compute_aucroc(true_A, self.A_tilde)

        return None
    
    def project_adjacency_matrix(self, projection_method):
        if projection_method == "rounding":
            idx_1 = torch.where(self.A_tilde >= 0.5)
            A_proj = torch.zeros_like(self.A_tilde)
            A_proj[idx_1] = 1.0
            return A_proj
        elif projection_method == "random":
            A_proj = torch.randint(0, 2, self.A_tilde.shape).double()
            A_proj = torch.triu(A_proj) + torch.triu(A_proj, 1).T
            A_proj.fill_diagonal_(0.0)
            return A_proj
        elif projection_method == "copy":
            return self.A_tilde.clone()
    
    def clip_adjacency_matrix(self, sigma_i):
        min_clip, max_clip = 0.0 - sigma_i, 1.0 + sigma_i
        return torch.clip(self.A_tilde, min_clip, max_clip)