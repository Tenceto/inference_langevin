import torch
from inspect import signature

import topology_inference.estimators_white_noise as est
import topology_inference.utils as ut


class AdamInitializer:
    def __init__(self, h_theta, theta_dist, lr, n_epochs):
        self.adam_est = est.AdamEstimator(h_theta=h_theta, theta_prior_dist=theta_dist, lr=lr, n_iter=n_epochs)
    
    def initial_estimation(self, Y, l1_penalty, num_runs, margin):
        A_adam_est = list()
        theta_adam_est = list()
        for i in range(num_runs):
            # Create a square matrix full of NaNs of the same shape as Y[0]
            A_nan = torch.full((Y.shape[0], Y.shape[0]), float('nan'))
            A_nan.fill_diagonal_(0.0)
            # Run the Adam estimator for a fully unknown graph
            A_adam, theta_adam, _ = self.adam_est.adam_estimate(A_nan=A_nan, Y=Y, l1_penalty=l1_penalty)
            A_adam_est.append(A_adam)
            theta_adam_est.append(theta_adam)

        A_adam = torch.stack(A_adam_est).mean(dim=0)
        theta_adam = torch.stack(theta_adam_est).mean(dim=0)

        A_initial_est = ut.threshold_probabilities(A_adam, margin)

        return A_initial_est, theta_adam


class BootstrapAdamInitializer:
    def __init__(self, h_theta, theta_dist, lr, n_epochs):
        self.adam_est = est.AdamEstimator(h_theta=h_theta, theta_prior_dist=theta_dist, lr=lr, n_iter=n_epochs)
    
    def initial_estimation(self, Y, l1_penalty, bootstrap_samples, margin):
        A_bootstrap_est = list()
        theta_bootstrap_est = list()
        for b in range(bootstrap_samples):
            # Create a square matrix full of NaNs of the same shape as Y[0]
            A_nan = torch.full((Y.shape[0], Y.shape[0]), float('nan'))
            A_nan.fill_diagonal_(0.0)
            # Bootstrap the data
            idx_bootstrap = torch.randint(0, Y.shape[1], (Y.shape[1],))
            Y_bootstrap = Y[:, idx_bootstrap]
            # Run the Adam estimator for a fully unknown graph
            A_adam, theta_adam, _ = self.adam_est.adam_estimate(A_nan=A_nan, Y=Y_bootstrap.to(A_nan.device), l1_penalty=l1_penalty)
            A_bootstrap_est.append(A_adam)
            theta_bootstrap_est.append(theta_adam)

        A_bootstrap = torch.stack(A_bootstrap_est).mean(dim=0)
        theta_bootstrap = torch.stack(theta_bootstrap_est).mean(dim=0)

        A_initial_est = ut.threshold_probabilities(A_bootstrap, margin)

        return A_initial_est, theta_bootstrap
