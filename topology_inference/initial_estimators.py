import torch

import topology_inference.estimators_white_noise as est
import topology_inference.utils as ut

torch.set_default_dtype(torch.float64)

class AdamInitializer:
    def __init__(self, h_theta, theta_dist, lr, n_epochs):
        self.adam_est = est.AdamEstimator(h_theta=h_theta, theta_prior_dist=theta_dist, lr=lr, n_iter=n_epochs)
    
    def initial_estimation(self, Y, l1_penalty, num_runs, margin):
        A_adam_est = list()
        theta_adam_est = list()
        for i in range(num_runs):
            # Create a square matrix full of NaNs of the same shape as Y[0]
            A_nan = torch.full((Y.shape[0], Y.shape[0]), float('nan'), device=Y.device)
            A_nan.fill_diagonal_(0.0)
            # Run the Adam estimator for a fully unknown graph
            A_adam, theta_adam, _ = self.adam_est.adam_estimate(A_nan=A_nan, Y=Y.to(A_nan.device), l1_penalty=l1_penalty)
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
            A_nan = torch.full((Y.shape[0], Y.shape[0]), float('nan'), device=Y.device)
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


class BootstrapSpectralInitializer:
    def __init__(self, h_theta, len_theta, epsilon_range=(0, 2), num_iter_reweight_refinements=10):
        self.spectral_estimator = est.SpectralTemplates()
        self.h_theta = h_theta
        self.len_theta = len_theta
        self.epsilon_range = epsilon_range
        self.num_iter_reweight_refinements = num_iter_reweight_refinements
    
    def initial_estimation(self, Y, threshold, bootstrap_samples, margin):
        A_bootstrap_est = list()
        theta_bootstrap_est = list()
        for b in range(bootstrap_samples):
            # Create a square matrix full of NaNs of the same shape as Y[0]
            A_nan = torch.full((Y.shape[0], Y.shape[0]), float('nan'), device=Y.device)
            A_nan.fill_diagonal_(0.0)
            # Bootstrap the data
            idx_bootstrap = torch.randint(0, Y.shape[1], (Y.shape[1],))
            Y_bootstrap = Y[:, idx_bootstrap]
            # Run the Spectral Templates estimator for a fully unknown graph
            emp_cov = (Y_bootstrap @ Y_bootstrap.T) / Y_bootstrap.shape[1]
            _, emp_cov_eigenvectors = torch.linalg.eigh(emp_cov)
            S_espectral, _, _ = self.spectral_estimator.spectral_templates(emp_cov=emp_cov.cpu().numpy(),
                                                                           emp_cov_eigenvectors=emp_cov_eigenvectors.cpu().numpy(),
                                                                           epsilon_range=self.epsilon_range,
                                                                           num_iter_reweight_refinements=self.num_iter_reweight_refinements)
            S_spectral_abs = torch.tensor(S_espectral).abs().fill_diagonal_(0.0)
            theta_spectral = self.spectral_estimator.lstsq_coefficients(S_spectral_abs.cuda(),
                                                                        Cx=emp_cov.cuda(),
                                                                        theta_length=self.len_theta, 
                                                                        threshold=threshold)
            A_spectral = (S_spectral_abs > threshold).double()

            A_bootstrap_est.append(A_spectral)
            theta_bootstrap_est.append(theta_spectral)

        A_bootstrap = torch.stack(A_bootstrap_est).mean(dim=0)
        theta_bootstrap = torch.stack(theta_bootstrap_est).mean(dim=0)

        A_initial_est = ut.threshold_probabilities(A_bootstrap, margin)

        return A_initial_est.to(Y.device), theta_bootstrap.to(Y.device)
