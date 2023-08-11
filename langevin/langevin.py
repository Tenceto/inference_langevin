import torch
from langevin.utils import compute_aucroc, compute_relative_error


class LangevinEstimator:
    def __init__(self, h_theta, A_score_model, sigma_e, theta_prior_dist=None, theta_fixed=None):
        self.h_theta = h_theta
        self.A_score_model = A_score_model
        self.sigma_e = sigma_e

        if theta_fixed is not None:
            self.theta_tilde = theta_fixed
            self.estimate_theta = False
        else:
            assert type(theta_prior_dist) is torch.distributions.normal.Normal, "Only normal prior is supported for theta."
            self.theta_prior_mean = theta_prior_dist.loc
            self.theta_prior_var = theta_prior_dist.scale ** 2
            self.estimate_theta = True
            # This distribution will be updated every time the noise level changes
            self.theta_prior_dist = None
        
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

    def langevin_estimate(self, A_nan, X, Y, sigmas_sq, epsilon, steps, temperature, projection_method="rounding", clip_A_tilde=False, true_A=None):
        # Initialize theta_tilde if it isn't fixed
        if self.estimate_theta:
            self.update_theta_annealed_prior(sigmas_sq[0])
            self.theta_tilde = self.theta_prior_dist.sample()

        unknown_idxs = torch.where(torch.isnan(torch.triu(A_nan)))
        known_mask = ~ torch.isnan(A_nan)
        size_unknown = len(unknown_idxs[0])
        
        dim_theta = len([self.theta_tilde])

        # Distributions for the Langevin noise (NOT the annealing one, these have unitary variance)
        z_dist = torch.distributions.MultivariateNormal(torch.zeros(size_unknown), torch.eye(size_unknown))
        v_dist = torch.distributions.MultivariateNormal(torch.zeros(dim_theta), torch.eye(dim_theta))

        # Initialize A_tilde and its projection
        self.A_tilde = torch.distributions.Normal(0.5, 0.1).sample(A_nan.shape)
        self.A_tilde = 0.5 * (torch.triu(self.A_tilde) + torch.triu(self.A_tilde, 1).T)
        self.A_tilde.fill_diagonal_(0.0)
        self.A_tilde[known_mask] = A_nan[known_mask]
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
                
                # Update A
                score_prior_A = self.A_score_model(self.A_tilde, sigma_i_idx)[unknown_idxs[0], unknown_idxs[1]]
                score_likelihood_A = self.score_joint_likelihood(X, Y, grad_variable="A")[unknown_idxs[0], unknown_idxs[1]]
                
                self.A_tilde[unknown_idxs[0], unknown_idxs[1]] = (self.A_tilde[unknown_idxs[0], unknown_idxs[1]]
                                                            + alpha * (score_prior_A + score_likelihood_A)
                                                            + torch.sqrt(2 * alpha * temperature) * z)
                self.A_tilde[unknown_idxs[1], unknown_idxs[0]] = self.A_tilde[unknown_idxs[0], unknown_idxs[1]]
                
                if clip_A_tilde:
                    self.A_tilde = self.clip_adjacency_matrix(torch.sqrt(sigma_i_sq))

                self.A_proj = self.project_adjacency_matrix(projection_method)

                # Update theta
                if self.estimate_theta:
                    score_prior_theta = self.compute_theta_prior_score()
                    score_likelihood_theta = self.score_joint_likelihood(X, Y, grad_variable="theta")
                    self.theta_tilde = self.theta_tilde + alpha * (score_prior_theta + score_likelihood_theta) + torch.sqrt(2 * alpha * temperature) * v

                # Compute metrics
                if compute_metrics:
                    self.metrics["aucroc"][steps * sigma_i_idx + t] = compute_aucroc(true_A, self.A_tilde, use_idxs=unknown_idxs)

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
    
    def update_theta_annealed_prior(self, sigma_i_sq):
        new_scale = torch.sqrt(self.theta_prior_var + sigma_i_sq)
        self.theta_prior_dist = torch.distributions.Normal(self.theta_prior_mean, new_scale)

class AdamEstimator:
    def __init__(self, h_theta, sigma_e, lr, n_iter):
        self.h_theta = h_theta
        self.sigma_e = sigma_e
        self.lr = lr
        self.n_iter = n_iter

    def adam_estimate(self, A_nan, X, Y):
        A_tilde = torch.distributions.Normal(0.5, 0.1).sample(A_nan.shape)
        A_tilde = 0.5 * (torch.triu(A_tilde) + torch.triu(A_tilde, 1).T)
        A_tilde.fill_diagonal_(0.0)
        
        unknown_mask = torch.isnan(A_nan)
        A_tilde[~ unknown_mask] = A_nan[~ unknown_mask]

        A_tilde.requires_grad_(True)
        loss_hist = []

        optimizer = torch.optim.Adam([A_tilde], lr=self.lr)

        for _ in range(self.n_iter):
            A = self.symmetrize_and_clip(A_tilde, A_nan, unknown_mask)
            optimizer.zero_grad()
            loss = self.compute_minus_likelihood(A, X, Y)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

        A = self.symmetrize_and_clip(A_tilde.detach(), A_nan, unknown_mask)
        
        return A, loss_hist
    
    def compute_minus_likelihood(self, A, X, Y):
        A_symmetric = torch.triu(A) + torch.triu(A, diagonal=1).T
        F = self.h_theta(A_symmetric)
        log_likelihood = - 1 / (2 * self.sigma_e ** 2) * (torch.linalg.norm(Y - F @ X, dim=0) ** 2).sum()
        return - log_likelihood
    
    def symmetrize_and_clip(self, A, A_nan, unknown_mask):
        A = torch.triu(A) + torch.triu(A, diagonal=1).T
        A = torch.clip(A, 0.0, 1.0)
        A[~ unknown_mask] = A_nan[~ unknown_mask]
        return A