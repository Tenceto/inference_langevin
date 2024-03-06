import torch
import cvxpy as cp
import numpy as np
from inspect import signature
from scipy.optimize import nnls


torch.set_default_dtype(torch.float64)

class LangevinEstimator:
    def __init__(self, h_theta, A_score_model, theta_prior_dist):
        self.h_theta = h_theta
        self.num_filter_params = len(signature(h_theta).parameters) - 1
        self.A_score_model = A_score_model
        # self.sigma_e = sigma_e
        self.theta_prior_dist = theta_prior_dist
        self.metrics = None

    def score_graph_likelihood(self, A, S, k, theta):
        A = A.clone().requires_grad_(True)
        # We only need the upper triangular part of A to account
        # for the symmetry of the matrix, so that the gradient is correct
        log_likelihood = - self.compute_minus_likelihood(torch.triu(A) + torch.triu(A, diagonal=1).T, 
                                                         S, k, theta)
        grad = torch.autograd.grad(log_likelihood, A)[0]
        return grad

    def compute_minus_likelihood(self, A, S, k, theta):
        F = self.h_theta(A, *theta)
        Omega = torch.linalg.inv(F @ F.T)
        log_likelihood = (torch.logdet(Omega) - torch.trace(S @ Omega)) * k / 2
        return - log_likelihood

    def langevin_estimate(self, A_nan, Y, sigmas_sq, steps, num_samples,
                          epsilon=1.0E-6, adam_lr=0.01, temperature=1.0,
                          projection_method="rounding", clip_A_tilde=False):

        k = Y.shape[1]
        S = (Y @ Y.T) / k

        As = []
        for _ in range(num_samples):
            this_A = self._generate_individual_sample(A_nan, S, k, sigmas_sq, steps, epsilon, adam_lr, temperature,
                                                      projection_method, clip_A_tilde)
            As.append(this_A)
        A = torch.stack(As).mean(dim=0)

        # Estimate the final theta from scratch using the final A
        theta_tilde = self.theta_prior_dist.sample([self.num_filter_params]).to(A_nan.device).abs()
        theta_tilde.requires_grad_(True)
        optimizer = torch.optim.Adam([theta_tilde], lr=adam_lr)
        for _ in range(steps * len(sigmas_sq)):
            optimizer.zero_grad()
            loss = self.compute_minus_likelihood(A, S, k, torch.clip(theta_tilde, min=1.0E-6))
            loss.backward()
            optimizer.step()
        theta = theta_tilde.detach()

        return A, theta

    def _langevin_individual_sample(self, A_nan, S, k, sigmas_sq, steps, epsilon, adam_lr, temperature,
                                    projection_method, clip_A_tilde):
        
        # Initialize theta_tilde
        theta_tilde = self.theta_prior_dist.sample([self.num_filter_params]).to(A_nan.device).abs()
        theta_tilde.requires_grad_(True)
        optimizer = torch.optim.Adam([theta_tilde], lr=adam_lr)

        unknown_idxs = torch.where(torch.isnan(torch.triu(A_nan)))
        known_mask = ~ torch.isnan(A_nan)
        size_unknown = len(unknown_idxs[0])
        # Distribution for the Langevin noise (NOT the annealing one, this one has unitary variance)
        z_dist = torch.distributions.MultivariateNormal(torch.zeros(size_unknown), torch.eye(size_unknown))

        # Initialize A_tilde and its projection
        A_tilde = torch.distributions.Normal(0.5, 0.1).sample(A_nan.shape).to(A_nan.device)
        A_tilde = 0.5 * (torch.triu(A_tilde) + torch.triu(A_tilde, 1).T)
        A_tilde.fill_diagonal_(0.0)
        A_tilde[known_mask] = A_nan[known_mask]
        A_proj = self.project_adjacency_matrix(A_tilde, projection_method)

        for sigma_i_idx, sigma_i_sq in enumerate(sigmas_sq):
            alpha = epsilon * sigma_i_sq / sigmas_sq[-1]
            for t in range(steps):
                z = z_dist.sample([1]).to(A_tilde.device)
                # Compute the score
                score_prior_A = self.A_score_model(A_tilde, sigma_i_idx)[unknown_idxs[0], unknown_idxs[1]]
                score_likelihood_A = self.score_graph_likelihood(A_tilde, S, k, 
                                                                 theta_tilde.clone().detach())[unknown_idxs[0], unknown_idxs[1]]
                # Update A
                A_tilde[unknown_idxs[0], unknown_idxs[1]] = (A_tilde[unknown_idxs[0], unknown_idxs[1]]
                                                             + alpha * (score_prior_A + score_likelihood_A)
                                                             + torch.sqrt(2 * alpha * temperature) * z)
                A_tilde[unknown_idxs[1], unknown_idxs[0]] = A_tilde[unknown_idxs[0], unknown_idxs[1]]
                if clip_A_tilde:
                    A_tilde = self.clip_adjacency_matrix(A_tilde, torch.sqrt(sigma_i_sq))
                A_proj = self.project_adjacency_matrix(A_tilde.clone().detach(), projection_method)
                # Update theta
                optimizer.zero_grad()
                loss = self.compute_minus_likelihood(A_proj, S, k, torch.clip(theta_tilde, min=1.0E-6))
                loss.backward()
                optimizer.step()

        return A_proj.detach()
    
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
    def __init__(self, h_theta, theta_prior_dist, lr, n_iter):
        self.h_theta = h_theta
        self.theta_prior_dist = theta_prior_dist
        # self.sigma_e = sigma_e
        self.lr = lr
        self.n_iter = n_iter
        self.num_filter_params = len(signature(h_theta).parameters) - 1

    def adam_estimate(self, A_nan, Y, l1_penalty):
        A_tilde = torch.distributions.Normal(0.5, 0.1).sample(A_nan.shape).to(A_nan.device)
        A_tilde = 0.5 * (torch.triu(A_tilde) + torch.triu(A_tilde, 1).T)
        A_tilde.fill_diagonal_(0.0)
        
        unknown_mask = torch.isnan(A_nan)
        A_tilde[~ unknown_mask] = A_nan[~ unknown_mask]

        A_known = A_tilde.clone()
        unknown_mask = unknown_mask.float()

        A_tilde.requires_grad_(True)
        theta = self.theta_prior_dist.sample([self.num_filter_params]).to(A_nan.device).abs()
        # if self.h_theta == heat_diffusion_filter:
        #     theta = theta.abs()
        theta.requires_grad_(True)
        optimizer = torch.optim.Adam([A_tilde, theta], lr=self.lr)
        loss_hist = []

        k = Y.shape[1]
        S = (Y @ Y.T) / k

        for _ in range(self.n_iter):
            A = self.symmetrize_and_clip(A_tilde)
            optimizer.zero_grad()
            loss = self.compute_penalized_minus_likelihood(A, Y, S, torch.clip(theta, min=1.0E-6),
                                                           l1_penalty, unknown_mask, A_known)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

        A = self.symmetrize_and_clip(A_tilde.detach())
        
        return A, theta.detach(), loss_hist
    
    def compute_penalized_minus_likelihood(self, A, Y, S, theta, l1_penalty, unknown_mask, A_known):
        A_symmetric = torch.triu(A) + torch.triu(A, diagonal=1).T
        A_symmetric = A * unknown_mask + A_known * (1 - unknown_mask)
        k = Y.shape[1]
        F = self.h_theta(A_symmetric, *theta)
        Omega = torch.linalg.inv(F @ F.T)
        log_likelihood = (torch.logdet(Omega) - torch.trace(S @ Omega)) * k / 2
        return - log_likelihood + l1_penalty * A_symmetric.norm(p=1)
        # return l1_penalty * A_symmetric.norm(p=1)
    
    # def compute_minus_likelihood(self, A, Y, S, theta, unknown_mask, A_known):
    #     A_symmetric = torch.triu(A) + torch.triu(A, diagonal=1).T
    #     A_symmetric = A * unknown_mask + A_known * (1 - unknown_mask)
    #     k = Y.shape[1]
    #     F = self.h_theta(A_symmetric, *theta)
    #     Omega = torch.linalg.inv(F @ F.T)
    #     log_likelihood = (torch.logdet(Omega) - torch.trace(S @ Omega)) * k / 2
    #     return - log_likelihood
    
    def symmetrize_and_clip(self, A):
        A = torch.triu(A) + torch.triu(A, diagonal=1).T
        A = torch.clip(A, 0.0, 1.0)
        return A


class SpectralTemplates:
    """
    Code copied from the original repository of the paper "pyGSL: A Graph Structure Learning Toolkit"
    by Max Wasserman and Gonzalo Mateos.
    """
    def __init__(self):
        pass

    def spectral_templates(self, emp_cov: np.ndarray, emp_cov_eigenvectors: np.ndarray, epsilon_range=(0, 2),
                           binary_search_iters: int=5,
                           tau=1, delta=.001, return_on_failed_iter_rew: bool=True,
                           num_iter_reweight_refinements: int=3,
                           verbose: bool=False):
        N = emp_cov.shape[-1]
        st_prob = self.spectral_template_problem(N, spec_temps_in=emp_cov_eigenvectors)
        st_param_dict, st_variable_dict = self._problem_dicts(st_prob)

        #############
        # Perform Binary Search on epsilon value. Resolve convex Spectral Templates problem for each epsilon.
        # Find smallest epsilon which allows a solution.
        epsilon_low, epsilon_high = epsilon_range
        # S_prev = solution to spectral temaples problem with smallest working epsilon
        smallest_working_epsilon, S_prev = None, None
        for i in range(binary_search_iters):
            epsilon = (epsilon_low + epsilon_high)/2
            st_param_dict['epsilon'].value = epsilon
            try:
                st_prob.solve(solver='MOSEK', warm_start=True, verbose=False)
                if st_prob.status == 'optimal':
                    worked = True
                    if verbose:
                        print(f'\tSpecTemp: {i}th iteration took: {st_prob.solver_stats.solve_time:.4f} s')#,  {st_prob.solver_stats.num_iters} iterations')
                else:
                    # infeasible, unbounded, etc
                    worked = False
                    if verbose:
                        print(f'\t{i}th binary search iteration failed: {st_prob.status}')
            except cp.error.SolverError as e:
                worked = False
                if verbose:
                    print(f'\t{i}th binary search iteration threw CVX exception: {e}')
            except Exception as e:
                worked = False
                if verbose:
                    print(f'\t{i}th binary search iteration threw OTHER exception: {e}')

            if worked:
                # worked, try smaller epsilon => smaller radius of Euclidean ball around S_hat
                epsilon_high = epsilon
                smallest_working_epsilon = epsilon
                S_prev = st_variable_dict['S'].value
            else:
                # didn't work, try larger epsilon => larger radius of Euclidean ball around S_hat
                epsilon_low = epsilon

        if S_prev is None:
            raise ValueError(f'\tNone of the epsilons in {epsilon_range} worked')

        #############
        # now apply iterative reweighting scheme a few times to clean up small edge weights.
        iter_rewt_prob = self.iterative_reweighted_problem(N=N, eps=st_param_dict['epsilon'].value, spec_temps_in=emp_cov_eigenvectors)
        iter_rewt_param_dict, iter_rewt_variable_dict = self._problem_dicts(iter_rewt_prob)

        worked = False
        for i in range(num_iter_reweight_refinements):
            iter_rewt_param_dict['weights'].value = self.compute_weights(S_prev=S_prev, tau=tau, delta=delta)
            # include try/except here for when solver fails. Better printing.
            try:
                iter_rewt_prob.solve(solver='MOSEK', warm_start=True, verbose=False)
                if iter_rewt_prob.status == 'optimal':
                    worked = True
                    if verbose:
                        print(f'\tIter Refine: {i}th iteration took: {iter_rewt_prob.solver_stats.solve_time:.4f} s')#,  {iter_rewt_prob.solver_stats.num_iters} iterations')
                else:
                    # infeasible, unbounded, etc
                    worked = False
                    if verbose:
                        print(f'\t{i}th Iterative Reweighting iteration failed: {iter_rewt_prob.status}')

            except cp.error.SolverError as e:
                worked = False
                if verbose:
                    print(f'\t{i}th Iterative Reweighting iteration threw CVX exception: {e}')
            except Exception as e:
                worked = False
                if verbose:
                    print(f'\t{i}th Iterative Reweighting iteration threw OTHER exception: {e}')

            if worked:
                S_prev = iter_rewt_variable_dict['S'].value
            elif return_on_failed_iter_rew:
                if i>0:
                    if verbose:
                        print(f'\t\tReturning {i - 1} Iterative Reweighting soln')
                else:
                    if verbose:
                        print(f'\t\tReturning Spectral Templates solution with NO Iterative Reweighting applied')

                return S_prev, smallest_working_epsilon, (i+1)

            else:
                raise ValueError(f'Iterative Reweighting Failed: '
                                f'To return last valid solution, set return_on_failed_iter_rew <- True')
            
        return S_prev, smallest_working_epsilon, num_iter_reweight_refinements
    
    @staticmethod
    def lstsq_coefficients(S_espectral, Cx, theta_length, threshold=0.5):
        S_espectral = S_espectral.to(Cx.device)
        A_spectral = (S_espectral.abs() > threshold).double()
        lam, V = torch.linalg.eigh(A_spectral)
        Cx_diag = torch.diag(V.T @ Cx @ V)
        a = torch.ones(len(lam), theta_length).cuda()
        for i in range(theta_length):
            a[:, i] = lam ** (theta_length - i - 1)
        b = Cx_diag.sqrt()
        # return torch.linalg.lstsq(a, b).solution
        solution = nnls(a.cpu().numpy(), b.cpu().numpy())[0]
        return torch.Tensor(solution).to(Cx.device)

    def spectral_template_problem(self, N, eps=None, spec_temps_in=None):
        # Define Variables and Parameters
        S_hat = cp.Variable((N, N), name='S_hat', symmetric=True)
        S = cp.Variable((N, N), name='S', symmetric=True)
        lam = cp.Variable(N)
        epsilon = cp.Parameter(nonneg=True, name='epsilon', value=eps)# if (eps != None) else None)
        spec_temps = cp.Parameter((N, N), 'eigenvectors', value=spec_temps_in)# if (spec_temps_in != None) else None)

        # Define objective and constraints
        objective = cp.Minimize(cp.sum(cp.abs(S)))  # cp.Minimize(cp.norm(S.flatten(), 1)) # CHECK CORRECTNESS: standard way to do sum of abs vals?
        constraints = [S_hat == spec_temps @ cp.diag(lam) @ spec_temps.T,
                    S >= 0,
                    cp.abs(cp.diag(S)) <= 1e-6,
                    S @ np.ones(N) >= 1,
                    cp.norm(S - S_hat, 'fro') <= epsilon]

        # Solve
        prob = cp.Problem(objective=objective, constraints=constraints)
        #assert prob.is_dcp(dpp=True), f'problem must comply with DPP rules for fast resolving.'
        return prob
    
    # for iterative_reweighted procedure
    def compute_weights(self, S_prev, tau, delta):
        ones_mat = np.ones_like(S_prev)
        weights_val = np.divide(tau * ones_mat, np.abs(S_prev) + delta * ones_mat)
        return weights_val
    
    def iterative_reweighted_problem(self, N, eps, spec_temps_in=None):
        # Define Variables and Parameters
        S_hat = cp.Variable((N, N), name='S_hat', symmetric=True) #this differs from matlab code. Dis he make mistake?
        S = cp.Variable((N, N), name='S', symmetric=True)
        lam = cp.Variable(N)
        epsilon = cp.Parameter(name='epsilon', nonneg=True, value=eps)
        spec_temps = cp.Parameter((N, N), name='eigenvectors', value=spec_temps_in)

        weights = cp.Parameter((N, N), name='weights', nonneg=True)

        # Define objective and constraints
        objective = cp.Minimize(cp.sum(cp.multiply(weights, S))) # elementwise multiply by weights
        constraints = [S_hat == spec_temps @ cp.diag(lam) @ spec_temps.T,
                    S >= 0,
                    cp.abs(cp.diag(S)) <= 1e-6,
                    S @ np.ones(N) >= 1,
                    cp.norm(S - S_hat, 'fro') <= epsilon]

        # Solve
        prob = cp.Problem(objective=objective, constraints=constraints)
        #assert prob.is_dcp(dpp=True), f'problem must comply with DPP rules for fast resolving.'
        return prob
    
    # getter function to access cvxpy problem parameters and attributes
    @staticmethod
    def _problem_dicts(problem):
        param_dict = {x.name(): x for x in problem.parameters()}
        variable_dict = {x.name(): x for x in problem.variables()}
        return param_dict, variable_dict