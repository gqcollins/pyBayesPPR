"""
Author: Gavin Collins
"""

import numpy as np
import scipy as sp
from scipy.special import comb, betaln
from scipy import stats
from itertools import combinations, chain
import time
from warnings import warn
import matplotlib.pyplot as plt
    
def get_move_type(n_ridge, n_quant, n_ridge_max):
    if n_ridge == 0:
      move_type = 'birth'
    elif n_ridge == n_ridge_max:
      if n_quant == 0:
        move_type = 'death'
      else:
        move_type = np.random.choice(['death', 'change'])
    else:
      if n_quant == 0:
        move_type = np.random.choice(['birth', 'death'])
      else:
        move_type = np.random.choice(['birth', 'death', 'change'])
    return move_type

def autocorrelation(chain, lag):
    """
    Calculate the autocorrelation of the chain at a specific lag.
    
    Parameters:
    - chain: 1D numpy array of numerical values.
    - lag: Integer specifying the lag at which to calculate autocorrelation.
    
    Returns:
    - autocorr: Autocorrelation value at the specified lag.
    """
    n = len(chain)
    mean = np.mean(chain)
    var = np.var(chain)
    
    # Ensure lag is non-negative and less than the length of the chain
    if lag < 0 or lag >= n:
        raise ValueError("Lag must be non-negative and less than the length of the chain.")
    
    # Calculate covariance
    cov = np.mean((chain[:n-lag] - mean) * (chain[lag:] - mean))
    
    # Normalize by variance, handling the case where variance is zero
    autocorr = cov / var if var > 0 else 0
    return autocorr

def effective_sample_size(chain):
    """
    Calculate the Effective Sample Size (ESS) for a given set of samples using numpy, with "chain" as the variable name.
    
    Parameters:
    - chain: Iterable of numerical values representing the samples.
    
    Returns:
    - ess: Effective Sample Size.
    """
    n = len(chain)
    if n < 2:
        return n
    
    # Check if variance is zero
    if np.var(chain) == 0:
        return 1  # Return ESS as 1 if there's no variability in the chain
    
    # Calculate autocorrelation for different lags
    max_lag = min(100, n // 2)  # Limiting the number of lags to check
    autocorrs = [autocorrelation(chain, lag) for lag in range(1, max_lag + 1)]
    
    # Sum the autocorrelations until they become non-positive
    sum_autocorrs = 0
    for autocorr in autocorrs:
        if autocorr <= 0:
            break
        sum_autocorrs += autocorr
    
    ess = n / (1 + 2 * sum_autocorrs)
    return ess

def split_chain_into_subchains(chain, n_subchains):
    """
    Split a single chain into multiple subchains.
    
    Parameters:
    - chain: A 1D numpy array or list representing the chain.
    - n_subchains: The number of subchains to split the chain into.
    
    Returns:
    - A 2D numpy array where each sub-array represents a subchain.
    """
    chain = np.array(chain)
    n_samples = len(chain)
    
    # Calculate the number of samples per subchain
    samples_per_subchain = n_samples // n_subchains
    
    # Trim the chain if necessary to make it evenly divisible
    trimmed_chain = chain[:samples_per_subchain * n_subchains]
    
    # Split the chain into subchains
    subchains = np.array_split(trimmed_chain, n_subchains)
    
    return np.array(subchains)

def calculate_rhat(chains):
    """
    Calculate the potential scale reduction factor, R-hat, for MCMC chains.
    
    Parameters:
    - chains: A 2D numpy array or list of lists where each sub-array/list represents a chain.
    
    Returns:
    - R-hat statistic for the chains.
    """
    chains = np.array(chains)  # Ensure chains is a numpy array for easier manipulation
    num_chains, num_samples = chains.shape
    
    # Step 1: Calculate the within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))
    
    # Step 2: Calculate the between-chain variance
    chain_means = np.mean(chains, axis=1)
    B = np.var(chain_means, ddof=1) * num_samples
    
    # Step 3: Estimate the marginal posterior variance
    V_hat = ((num_samples - 1) / num_samples) * W + (1 / num_samples) * B
    
    # Step 4: Calculate R-hat
    R_hat = np.sqrt(V_hat / W)
    
    return R_hat

def lchoose(n, k):
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)
        
def combn(n, k):
    """Get all combinations of indices from 0:n of length k"""
    # https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
    count = comb(n, k, exact=True)
    idx = np.fromiter(chain.from_iterable(combinations(range(n), k)), int, count=count * k)
    return idx.reshape(-1, k)

def dwallenius(w_feat_norm, feat):
    """Multivariate Walenius' noncentral hypergeometric density function with some variables fixed"""
    if(len(feat) == len(w_feat_norm)):
        return 1.0
    else:
        logMH = w_feat_norm[feat] / (1 - np.sum(w_feat_norm[feat]))
        j = len(logMH)
        ss = 1 + (-1)**j / (np.sum(logMH) + 1)
        for i in range(1, j):
            idx = combn(j, i)
            ss += (-1)**i * np.sum(1 / (np.sum(logMH[idx], axis=1) + 1))
        return ss

def get_cat_basis(Xj):
    p = np.shape(Xj)[1]
    if p > 1:
        tp = (1 - Xj[:, 0])
        for j in range(1,p):
            tp *= (1 - Xj[:, j])
        basis = 1 - tp
    else:
        basis = Xj.copy()
    return basis

def rps(mu, kappa): # Get a random draw from the "power spherical" distribution, which has density f(x) \propto (1 + t(mu) %*% x)^kappa
    d = len(mu)
    
    uhat = -mu
    uhat[0] = uhat[0] + 1
    u = uhat / np.sqrt(np.sum(uhat**2))
    
    b = (d - 1)/2
    a = b + kappa
    z = np.random.beta(a, b)
    t = 2*z - 1
    
    temp = np.random.normal(size = d - 1)
    v = temp / np.sqrt(np.sum(temp**2))
    y = np.append([t], np.sqrt(1 - t**2) * v)
    
    uy = np.sum(u * y)
    x = y - 2 * u * uy
    return x

def relu(x): # Get relu of input
    return np.reshape((np.abs(x) + x) / 2, (len(x), 1))

def get_mns_basis(u, knots): # Get modified natural spline basis
    n_knots = len(knots)
    df = n_knots - 2
    
    basis = relu(u - knots[0]) # Initialize basis -- would be basis = u for ns
    
    if df > 1:
      n_internal_knots = n_knots - 3
      r = []
      d = []
      for k in range(1,n_knots):
        r.append(relu(u - knots[k])**3)
      for k in range(df):
        d.append((r[k] - r[df]) / (knots[df + 1] - knots[k + 1]))
      for k in range(n_internal_knots):
        basis = np.hstack([basis, d[k] - d[n_internal_knots]])
    return basis

def get_log_mh_bd(n_ridge_prop, n_quant_prop, n_ridge_max):
    if n_ridge_prop == 0:
        alpha0 = 0
    elif n_ridge_prop == n_ridge_max:
        if n_quant_prop == 0:
            alpha0 = 0
        else:
            alpha0 = np.log(2)
    else:
        if n_quant_prop == 0:
            alpha0 = np.log(2)
        else:
            alpha0 = np.log(3)

    return alpha0

class qf_info:
    def __init__(self, BtB, Bty):
        """Get the quadratic form y'B solve(B'B) B'y, as well as least squares coefs and cholesky of B'B"""
        self.dim = None
        self.chol = None
        self.ls_est = None
        self.qf = None
        
        try:
            chol_BtB = sp.linalg.cholesky(BtB, lower=False)  # might be a better way to do this with sp.linalg.cho_factor
        except (np.linalg.LinAlgError, ValueError):
            return
        d = np.diag(chol_BtB)
        dim = len(d)
        if dim > 1:
            if np.max(d[1:]) / np.min(d) > 1e3:
                return
        
        self.dim = dim
        self.chol = chol_BtB
        self.ls_est = sp.linalg.solve_triangular(self.chol, sp.linalg.solve_triangular(self.chol, Bty, trans=1))
        self.qf = Bty.T @ self.ls_est
        return
    
    def get_inv_chol(self):
        self.inv_chol = sp.linalg.solve_triangular(self.chol, np.identity(self.dim))
        return
        
    
class bpprPrior:
    def __init__(self, n_ridge_mean, n_ridge_max, n_act_max, df_spline,
                 prob_relu, prior_coefs, shape_var_coefs, scale_var_coefs,
                 n_dat_min
                 ):    
        assert prior_coefs in ['zs', 'flat'], "prior_coefs must be either 'zs' or 'flat'"
        self.n_ridge_mean = n_ridge_mean
        self.n_ridge_max = n_ridge_max
        self.n_act_max = n_act_max
        self.df_spline = df_spline
        self.knot_quants = np.linspace(0.0, 1.0, num=self.df_spline + 1)
        self.prob_relu = prob_relu
        self.prior_coefs = prior_coefs
        self.shape_var_coefs = shape_var_coefs
        self.scale_var_coefs = scale_var_coefs
        self.n_dat_min = n_dat_min
        return
    
    def calibrate(self, data): # Maximum proportion of inactive datapoints in a ridge function
        if self.prior_coefs == 'zs':
            if self.shape_var_coefs is None:
                self.shape_var_coefs = 0.5
            if self.scale_var_coefs is None:
                self.scale_var_coefs = data.n/2
                
        if self.n_dat_min is None:
            self.n_dat_min = min(20, 0.1 * data.n)
        if self.n_dat_min <= self.df_spline:
            warn('n_dat_min too small. If n_dat_min was set by default, df_spline is large compared to the sample size. Setting n_dat_min = df_spline + 1')
            self.n_dat_min = self.df_spline + 1
        self.p_dat_max = 1.0 - self.n_dat_min / data.n 
        
        if self.n_act_max is None:
            n_cat = np.sum(data.feat_type == np.repeat('cat', data.p))
            self.n_act_max = int(min(3, data.p - n_cat) + min(3, np.ceil(n_cat/2)))
        
        self.proj_dir_mn = [np.repeat(1/np.sqrt(a), a) for a in range(1, self.n_act_max+1)] # prior mean for proj_dir (arbitrary, since prior precision is zero)
        
        if self.n_ridge_max is None:
            self.n_ridge_max = int(min(150, np.floor(data.n/self.df_spline) - 2))
        assert self.n_ridge_max > 0, 'n_ridge_max <= 0. If n_ridge_max was set by default, df_spline is too large compared to the sample size.'
        return
        
    
class bpprData:
    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        return

    def summarize(self, prior):
        self.n, self.p = np.shape(self.X)
        self.ssy = self.y.T @ self.y
        self.mn_X = np.zeros(self.p)
        self.sd_X = np.ones(self.p)
        self.feat_type = []
        for j in range(self.p):
            n_unique = len(set(self.X[:, j]))
            if n_unique == 1:
                self.feat_type.append('')
            elif n_unique == 2:
                self.feat_type.append('cat')
            else:
                self.mn_X[j] = np.mean(self.X[:, j])
                self.sd_X[j] = np.std(self.X[:, j], ddof=1)
                if n_unique <= prior.df_spline:
                    self.feat_type.append('disc')
                else:
                    self.feat_type.append('cont')
        self.feat_type = np.array(self.feat_type)
        self.standardize()
        return
        
    def standardize(self, X=None):
        if X is None: # get standardized version of self.X
            self.X_st = self.X.copy()
            for j in range(self.p):
                self.X_st[:, j] = (self.X[:, j] - self.mn_X[j]) / self.sd_X[j]
            return
        else: # return standardized version of X
            X_st = X.copy()
            for j in range(self.p):
                X_st[:, j] = (X[:, j] - self.mn_X[j]) / self.sd_X[j]
            return X_st
            

class bpprSpecs:
    def __init__(self, n_post, n_burn, n_adapt, n_thin,
                 w_n_act, w_feat, adapt_act_feat, scale_proj_dir_prop
                 ):
        assert n_thin <= n_post, 'n_thin > n_post. No posterior samples will be obtained.'
        self.n_post = n_post
        self.n_burn = n_burn
        self.n_adapt = n_adapt
        self.n_thin = n_thin
        self.n_post -= self.n_post % self.n_thin
        self.n_keep = int(np.floor(self.n_post/self.n_thin))
        self.n_pre = self.n_adapt + self.n_burn
        self.n_draws = self.n_pre + self.n_post
        
        if w_n_act is None:
            self.w_n_act = None # assign later
        else:
            self.w_n_act = w_n_act.copy()
        if w_feat is None:
            self.w_feat = None # assign later
        else:
            self.w_feat = w_feat.copy()
        self.adapt_act_feat = adapt_act_feat
        
        if scale_proj_dir_prop is None:
            self.proj_dir_prop_prec = 1000.0  # scale_proj_dir_prop = 0.002
        else:
            assert scale_proj_dir_prop > 0 and scale_proj_dir_prop <= 1, "scale_proj_dir_prop must be in (0, 1]"
            inv_scale_proj_dir_prop = 1/scale_proj_dir_prop
            self.proj_dir_prop_prec = (inv_scale_proj_dir_prop - 1) + np.sqrt(inv_scale_proj_dir_prop * (inv_scale_proj_dir_prop - 1))
        return
    
    def calibrate(self, data, prior):
        for j,ft in enumerate(data.feat_type):
            if ft == '':
                self.w_feat[j] == 0.0
        
        if prior.prior_coefs == 'flat':
            self.n_adapt = self.n_pre
            self.n_burn = 0
        
        if self.w_n_act is None:
            self.w_n_act = np.ones(prior.n_act_max) 
        
        if self.w_feat is None:
            self.w_feat = np.ones(data.p)  # assign later
        return
    
    
class bpprBirthProposal:
    def __init__(self, state, data, prior, specs):
        self.n_ridge = state.n_ridge + 1
        # Propose number of active features
        self.n_act = np.random.choice(prior.n_act_max, p=state.w_n_act_norm) + 1
        if specs.adapt_act_feat:
            # Nott, Kuk, Duc for n_act
            state.log_mh_act_feat = -(np.log(prior.n_act_max) + np.log(state.w_n_act_norm[self.n_act-1]))
            # Propose features to include
            if self.n_act == 1:
                self.feat = np.random.choice(data.p, 1)
            else:
                self.feat = np.random.choice(data.p, self.n_act, p=state.w_feat_norm, replace=False)
            if self.n_act > 1  and  self.n_act < prior.n_act_max:
                state.log_mh_act_feat -= lchoose(data.p, self.n_act) + np.log(dwallenius(state.w_feat_norm, self.feat))  # Nott, Kuk, Duc for feat
        else:
            # Propose features to include
            self.feat = np.random.choice(data.p, self.n_act, p=state.w_feat_norm, replace = False)

        if np.all(data.feat_type[self.feat] == 'cat'):  # Are all of the proposed features categorical?
            self.ridge_type = 'cat'
            self.n_quant = state.n_quant
            self.proj_dir = [np.nan]
            self.knots = [np.nan]
            self.ridge_basis = get_cat_basis(data.X_st[:, self.feat])
            self.n_basis = 1
        else:
            self.n_quant = state.n_quant + 1
            if self.n_act == 1:
                self.proj_dir = np.random.choice([-1, 1], 1)
            else:
                # Propose direction
                self.proj_dir = rps(prior.proj_dir_mn[self.n_act - 1], 0.0)
            self.proj = data.X_st[:, self.feat] @ self.proj_dir # Get proposed projection
            
            if np.any(data.feat_type[self.feat] == 'cont'):  # Are any proposed features continuous?
                self.ridge_type = 'cont'
                max_knot0 = np.quantile(self.proj, prior.p_dat_max)
                rg_knot0 = (max_knot0 - np.min(self.proj)) / prior.prob_relu
                knot0 = max_knot0 - rg_knot0 * np.random.uniform()
                self.knots = np.append([knot0], np.quantile(self.proj[self.proj > knot0], prior.knot_quants))  # Get proposed knots
                # Get proposed basis functions
                self.ridge_basis = get_mns_basis(self.proj, self.knots)
                self.n_basis = prior.df_spline
                # The proposed features are a mix of categorical and discrete quantitative
            else:
                self.ridge_type = 'disc'
                self.knots = [np.nan]
                self.ridge_basis = self.proj.copy()
                self.n_basis = 1

        # inner product of proposed new basis functions
        self.PtP = self.ridge_basis.T @ self.ridge_basis
        self.BtP = state.basis_mat.T @ self.ridge_basis
        self.Pty = self.ridge_basis.T @ data.y
        
        self.idx_ridge = state.n_ridge
        basis_idx_start = state.basis_idx[self.idx_ridge].stop
        self.basis_idx = slice(basis_idx_start, basis_idx_start + self.n_basis)
        return
    
    def get_log_mh(self, state, data, prior):
        self.qf_info = qf_info(state.BtB[:(state.n_basis_total + self.n_basis), :(state.n_basis_total + self.n_basis)],
                               state.Bty[:(state.n_basis_total + self.n_basis)])

        self.log_mh = None
        if self.qf_info.qf is not None:
            if self.qf_info.qf < data.ssy:
                self.log_mh_bd = get_log_mh_bd(self.n_ridge, self.n_quant, prior.n_ridge_max)

                self.sse = data.ssy - state.c_var_coefs * self.qf_info.qf

                # Compute the acceptance probability
                self.log_mh = (
                    state.log_mh_bd - self.log_mh_bd + state.log_mh_act_feat + # Adjustment for probability of birth proposal
                    -data.n/2 * (np.log(self.sse) - np.log(state.sse)) + # Part of the marginal likelihood
                    np.log(prior.n_ridge_mean/(state.n_ridge + 1)) # Prior and proposal distribution
                    )
                if prior.prior_coefs == 'zs':
                    # The rest of the marginal likelihood for Zellner-Siow prior
                    self.log_mh -= self.n_basis * np.log(state.var_coefs + 1)/2
                else:
                    # The rest of the marginal likelihood for flat prior
                    self.log_mh += np.log(10e-6)
        return
    
    
class bpprDeathProposal:
    def __init__(self, state, data, prior, specs):
        # Choose random index to delete
        self.idx_ridge = np.random.choice(state.n_ridge)

        self.n_act = state.n_act[self.idx_ridge]
        self.feat = state.feat[self.idx_ridge, :self.n_act]
        if specs.adapt_act_feat:
            state.log_mh_act_feat = np.log(prior.n_act_max) + np.log((state.w_n_act[self.n_act-1] - 1)/(np.sum(state.w_n_act) - 1))
            self.w_feat = state.w_feat.copy()
            self.w_feat[self.feat] -= 1
            self.w_feat_norm = self.w_feat/np.sum(self.w_feat)
            if self.n_act > 1:
                state.log_mh_act_feat += lchoose(data.p, self.n_act) + np.log(dwallenius(self.w_feat_norm, self.feat))  # Nott, Kuk, and Duc

        self.idx_basis = list(range(state.n_basis_total))
        del self.idx_basis[state.basis_idx[self.idx_ridge + 1]]     
        
        self.n_ridge = state.n_ridge - 1
        if state.ridge_type[self.idx_ridge] == 'cat':
            self.n_quant = state.n_quant
        else:
            self.n_quant = state.n_quant - 1
        return
    
    def get_log_mh(self, state, data, prior):
        self.qf_info = qf_info(state.BtB[np.ix_(self.idx_basis, self.idx_basis)],
                               state.Bty[self.idx_basis])
        
        self.log_mh = None
        if self.qf_info.qf is not None:
            if self.qf_info.qf < data.ssy:
                self.log_mh_bd = get_log_mh_bd(self.n_ridge, self.n_quant, prior.n_ridge_max)

                self.sse = data.ssy - state.c_var_coefs * self.qf_info.qf
                self.n_basis = state.n_basis_ridge[self.idx_ridge + 1]

                # Compute acceptance probability
                self.log_mh = (state.log_mh_bd - self.log_mh_bd + state.log_mh_act_feat + # Adjustment for probability of death proposal
                               -data.n/2 * (np.log(self.sse) - np.log(state.sse)) + # Part of the marginal likelihood
                               np.log(state.n_ridge/prior.n_ridge_mean) # Prior and proposal distribution
                               )
                
                if prior.prior_coefs == 'zs':
                    # The rest of the marginal likelihood for Zellner-Siow prior
                    self.log_mh += 0.5 * self.n_basis * np.log(state.var_coefs + 1)
                else:
                    # The rest of the marginal likelihood for flat prior
                    self.log_mh -= np.log(10e-6)
        return
    
    
class bpprChangeProposal:
    def __init__(self, state, data, prior, specs):
        self.idx_ridge = np.random.choice(state.idx_ridge_quant) # Which ridge function should we change?
        
        self.n_act = state.n_act[self.idx_ridge].copy()
        self.feat = state.feat[self.idx_ridge][:self.n_act].copy()
        proj_dir_curr = state.proj_dir[self.idx_ridge][:self.n_act].copy()
        
        # Get proposed direction
        if state.n_act[self.idx_ridge] == 1:
            self.proj_dir = np.random.choice([-1, 1], 1)
        else:
            self.proj_dir = rps(proj_dir_curr, specs.proj_dir_prop_prec)

        self.proj = data.X_st[:,self.feat] @ self.proj_dir # Get proposed projection

        if state.ridge_type[self.idx_ridge] == 'cont':  # Are any variables continuous for this ridge function?
            max_knot0 = np.quantile(self.proj, prior.p_dat_max)
            rg_knot0 = (max_knot0 - np.min(self.proj)) / prior.prob_relu
            knot0 = max_knot0 - rg_knot0 * np.random.uniform()
            self.knots = np.append([knot0], np.quantile(self.proj[self.proj > knot0], prior.knot_quants))  # Get proposed knots
            self.ridge_basis = get_mns_basis(self.proj, self.knots) # Get proposed basis function
        else:
            self.knots = [np.nan]
            self.ridge_basis = self.proj.copy()

        # inner product of proposed new basis functions
        PtP = self.ridge_basis.T @ self.ridge_basis
        BtP = state.basis_mat.T @ self.ridge_basis
        Pty = self.ridge_basis.T @ data.y

        self.BtB = state.BtB[:state.n_basis_total, :state.n_basis_total].copy()
        self.BtB[state.basis_idx[self.idx_ridge + 1], :] = BtP.T.copy()
        self.BtB[:, state.basis_idx[self.idx_ridge + 1]] = BtP.copy()
        self.BtB[state.basis_idx[self.idx_ridge + 1], state.basis_idx[self.idx_ridge + 1]] = PtP.copy()

        self.Bty = state.Bty[:state.n_basis_total].copy()
        self.Bty[state.basis_idx[self.idx_ridge + 1]] = Pty.copy()
        return
    
    def get_log_mh(self, state, data, prior):
        self.qf_info = qf_info(self.BtB, self.Bty)
        
        self.log_mh = None
        if self.qf_info.qf is not None:
            if self.qf_info.qf < data.ssy:
                self.sse = data.ssy - state.c_var_coefs * self.qf_info.qf

                # Compute the acceptance probability
                self.log_mh = -data.n/2 * (np.log(self.sse) - np.log(state.sse))  # Marginal Likelihood
        return
    
    
class bpprState:
    def __init__(self, data, prior, specs):
        # Current value of parameters
        self.n_ridge = 0  # Number of ridge functions
        self.n_act = np.zeros(prior.n_ridge_max, dtype=int) # Number of active features for jth ridge function
        self.feat = np.zeros([prior.n_ridge_max, prior.n_act_max], dtype=int)  # Features being used in jth ridge function
        self.proj_dir = np.zeros([prior.n_ridge_max, prior.n_act_max])  # Ridge directions
        self.knots = np.zeros([prior.n_ridge_max, prior.df_spline + 2]) # Location of knots for nsplines
        self.coefs = np.zeros([prior.df_spline * prior.n_ridge_max + 1]) # Basis coefficients
        self.coefs[0] = np.mean(data.y)
        self.s2 = 1.0 # Error variance
        if prior.prior_coefs == 'zs':
            self.var_coefs = prior.scale_var_coefs / prior.shape_var_coefs
            self.c_var_coefs = self.var_coefs / (self.var_coefs + 1.0)
        elif prior.prior_coefs == 'flat':
            self.var_coefs = np.nan
            self.c_var_coefs = 1.0        
           
        # Other things to track
        self.n_basis_ridge = [1] # Number of basis functions in each ridge function (first ridge function is the intercept)
        self.n_basis_total = np.sum(self.n_basis_ridge)
        self.ridge_type = []
        self.idx_ridge_quant = []  # which ridge functions are quantitative?
        self.n_quant = 0  # how many ridge functions are quantitative?
        self.basis_mat = np.ones((data.n, 1))  # Current basis matrix
        self.basis_idx = [slice(0,1)]  # Current indices of segments of basis functions
        self.BtB = np.zeros((prior.n_ridge_max * prior.df_spline + 1,
                             prior.n_ridge_max * prior.df_spline + 1))
        self.BtB[:self.n_basis_total, :self.n_basis_total] = self.basis_mat.T @ self.basis_mat
        self.Bty = np.zeros(prior.n_ridge_max * prior.df_spline + 1)
        self.Bty[:self.n_basis_total] = self.basis_mat.T @ data.y
        self.qf_info = qf_info(self.BtB[:self.n_basis_total, :self.n_basis_total],
                               self.Bty[:self.n_basis_total])
        self.qf_info.get_inv_chol()

        self.sse = data.ssy - self.c_var_coefs * self.qf_info.qf
        self.log_mh_bd = 0.0
        self.log_mh_act_feat = 0.0
        
        self.idx = 0
        if specs.n_adapt > 0:
            self.phase = 'adapt'
        elif specs.n_burn > 0:
            self.phase = 'burn'
        else:
            self.phase = 'post-burn'
            
        self.w_n_act = specs.w_n_act.copy()
        self.w_n_act_norm = self.w_n_act / np.sum(self.w_n_act)
        self.w_feat = specs.w_feat.copy()
        self.w_feat_norm = self.w_feat / np.sum(self.w_feat)
        return
    
    def acceptBirth(self, prop, adapt_act_feat):
        self.n_ridge += 1
        self.n_act[prop.idx_ridge] = prop.n_act
        self.feat[prop.idx_ridge, :prop.n_act] = prop.feat.copy()
        self.knots[prop.idx_ridge, :] = prop.knots.copy()
        self.proj_dir[prop.idx_ridge, :prop.n_act] = prop.proj_dir.copy()

        if prop.ridge_type != 'cat':
            self.idx_ridge_quant.append(prop.idx_ridge)
            self.n_quant = prop.n_quant
        self.ridge_type.append(prop.ridge_type)

        self.basis_idx.append(prop.basis_idx)
        self.n_basis_ridge.append(prop.n_basis)
        self.n_basis_total += prop.n_basis
        self.basis_mat = np.append(self.basis_mat, prop.ridge_basis, axis=1)

        # Update weights
        if adapt_act_feat:
            self.w_n_act[prop.n_act-1] += 1
            self.w_n_act_norm = self.w_n_act/np.sum(self.w_n_act)
            self.w_feat[prop.feat] += 1
            self.w_feat_norm = self.w_feat/np.sum(self.w_feat)

        self.qf_info = prop.qf_info
        if self.phase != 'adapt':
            self.qf_info.get_inv_chol()
        self.sse = prop.sse
        self.log_mh_bd = prop.log_mh_bd
        return
    
    def acceptDeath(self, prop, adapt_act_feat):
        self.n_basis_total -= prop.n_basis
        self.n_ridge -= 1
        if self.ridge_type[prop.idx_ridge] != 'cat':
            self.n_quant -= 1
            self.idx_ridge_quant.remove(prop.idx_ridge)

        for k in range(len(self.idx_ridge_quant)):
            if self.idx_ridge_quant[k] > prop.idx_ridge:
                self.idx_ridge_quant[k] -= 1
            
        del self.ridge_type[prop.idx_ridge]
        
        self.basis_mat = np.delete(self.basis_mat, self.basis_idx[prop.idx_ridge+1], axis=1)
        self.BtB[:self.n_basis_total, :self.n_basis_total] = self.BtB[np.ix_(prop.idx_basis, prop.idx_basis)].copy()
        self.Bty[:self.n_basis_total] = self.Bty[prop.idx_basis].copy()

        for j in range(prop.idx_ridge, self.n_ridge):
            self.n_act[j] = self.n_act[j + 1].copy()
            self.feat[j] = self.feat[j + 1].copy()
            self.knots[j] = self.knots[j + 1].copy()
            self.proj_dir[j] = self.proj_dir[j + 1].copy()
            self.basis_idx[j + 1] = slice(self.basis_idx[j + 2].start - self.n_basis_ridge[prop.idx_ridge + 1],
                                          self.basis_idx[j + 2].stop - self.n_basis_ridge[prop.idx_ridge + 1]
                                          )

        del self.basis_idx[-1]
        del self.n_basis_ridge[prop.idx_ridge + 1]

        # Update weights
        if adapt_act_feat:
            self.w_feat = prop.w_feat.copy()
            self.w_feat_norm = prop.w_feat_norm.copy()
            self.w_n_act[prop.n_act-1] -= 1
            self.w_n_act_norm = self.w_n_act/np.sum(self.w_n_act)

        self.qf_info = prop.qf_info
        if self.phase != 'adapt':
            self.qf_info.get_inv_chol()
        self.sse = prop.sse
        self.log_mh_bd = prop.log_mh_bd
        return
    
    def acceptChange(self, prop):
        self.knots[prop.idx_ridge, :] = prop.knots.copy()
        self.proj_dir[prop.idx_ridge, :prop.n_act] = prop.proj_dir.copy()

        self.BtB[:self.n_basis_total, :self.n_basis_total] = prop.BtB.copy()
        self.Bty[:self.n_basis_total] = prop.Bty.copy()
        self.basis_mat[:, self.basis_idx[prop.idx_ridge + 1]] = prop.ridge_basis.copy()

        self.qf_info = prop.qf_info
        if self.phase != 'adapt':
            self.qf_info.get_inv_chol()
        self.sse = prop.sse
        return
    
    def sampleSDResid(self, data):
        self.s2 = 1/np.random.gamma(data.n/2, 2/self.sse)
        return
    
    def sampleCoefs(self):
        self.coefs[:self.n_basis_total] = (self.c_var_coefs * self.qf_info.ls_est +
                                          np.sqrt(self.c_var_coefs * self.s2) * self.qf_info.inv_chol @ np.random.normal(size = self.n_basis_total)
                                          )
        return
    
    def sampleVarCoefs(self, data, prior):
        qf_comp = self.qf_info.chol @ self.coefs[:self.n_basis_total]
        qf = qf_comp.T @ qf_comp
        
        self.var_coefs = 1/np.random.gamma(prior.shape_var_coefs + self.n_basis_total/2,
                                           1 / (prior.scale_var_coefs + qf/(2*self.s2))
                                           )
        self.c_var_coefs = self.var_coefs / (self.var_coefs + 1)
        self.sse = data.ssy - self.c_var_coefs * self.qf_info.qf
        return
    
    def update(self, data, prior, specs):
        move_type = get_move_type(self.n_ridge, self.n_quant, prior.n_ridge_max)
        
        if move_type == 'birth':  
            # Generate birth proposal
            prop = bpprBirthProposal(self, data, prior, specs)
            
            # update quadratic forms just in case proposal is accepted
            self.BtB[:self.n_basis_total, prop.basis_idx] = prop.BtP.copy()
            self.BtB[prop.basis_idx, :self.n_basis_total] = prop.BtP.T.copy()
            self.BtB[prop.basis_idx, prop.basis_idx] = prop.PtP.copy()
            self.Bty[prop.basis_idx] = prop.Pty.copy()
            
            # Calculate log(MH acceptance probability)
            prop.get_log_mh(self, data, prior)
    
            if prop.log_mh is not None:
                if np.log(np.random.uniform()) < prop.log_mh:
                    self.acceptBirth(prop, specs.adapt_act_feat)
    
        elif move_type == 'death':  # Death step
            # Generate death proposal
            prop = bpprDeathProposal(self, data, prior, specs)
            
            # Calculate log(MH acceptance probability)
            prop.get_log_mh(self, data, prior)
    
            if prop.log_mh is not None:
                if np.log(np.random.uniform()) < prop.log_mh:
                    self.acceptDeath(prop, specs.adapt_act_feat)
    
        else:  # Change Step
            # Generate change proposal
            prop = bpprChangeProposal(self, data, prior, specs)
            
            # Calculate log(MH acceptance probability)
            prop.get_log_mh(self, data, prior)
    
            if prop.log_mh is not None:
                if np.log(np.random.uniform()) < prop.log_mh:
                    self.acceptChange(prop)
    
        if self.phase != 'adapt':
            self.sampleSDResid(data)
    
            self.sampleCoefs()
    
            if prior.prior_coefs == 'zs':
                self.sampleVarCoefs(data, prior)
        return
    
    
class bpprSamples:
    def __init__(self, prior, specs, state0=None):
        self.n_ridge = np.zeros(specs.n_keep, dtype=int)  # Number of ridge functions
        self.ridge_type = [None] * specs.n_keep
        self.n_act = np.zeros([specs.n_keep, prior.n_ridge_max], dtype=int) # Number of active features for jth ridge function
        self.feat = np.zeros([specs.n_keep, prior.n_ridge_max, prior.n_act_max], dtype=int)  # Features being used in jth ridge function
        self.proj_dir = np.zeros([specs.n_keep, prior.n_ridge_max, prior.n_act_max])  # Ridge directions
        self.knots = np.zeros([specs.n_keep, prior.n_ridge_max, prior.df_spline + 2]) # Location of knots for nsplines
        self.coefs = np.zeros([specs.n_keep, prior.df_spline * prior.n_ridge_max + 1]) # Basis coefficients
        self.s2 = np.zeros(specs.n_keep) # Error variance
        if prior.prior_coefs == 'zs':
            self.var_coefs = np.zeros(specs.n_keep)
        elif prior.prior_coefs == 'flat':
            self.var_coefs = None
            
        if state0 is not None:
            self.writeState(state0)
        return
    
    def writeState(self, state):
        self.n_ridge[state.idx] = state.n_ridge
        self.ridge_type[state.idx] = state.ridge_type.copy()
        self.n_act[state.idx] = state.n_act.copy()
        self.feat[state.idx] = state.feat.copy()
        self.proj_dir[state.idx] = state.proj_dir.copy()
        self.knots[state.idx] = state.knots.copy()
        self.coefs[state.idx] = state.coefs.copy()
        self.s2[state.idx] = state.s2
        if state.var_coefs is not None:
            self.var_coefs[state.idx] = state.var_coefs
        return


class bpprModel:
    """The model structure, including the current RJMCMC state and previous saved states; with methods for saving the
        state, plotting MCMC traces, and predicting"""
    def __init__(self, data, prior, specs, samples):
        self.data = data
        self.prior = prior
        self.specs = specs
        self.samples = samples
        return

    def predict(self, newdata, mcmc_use=None):
        n, p = np.shape(newdata)

        newdata_s = self.data.standardize(newdata)
        
        if mcmc_use is None:
            mcmc_use = np.array(range(self.specs.n_keep))
        else:
            assert max(mcmc_use) <= self.specs.n_keep, "invalid 'mcmc_use'"
        n_use = len(mcmc_use)
        
        ridge_basis = [None] * np.max(self.samples.n_ridge) 
        preds = np.zeros((n_use, n))
        for i in range(n_use):
            preds[i] = self.samples.coefs[mcmc_use[i], 0]
            calc_all_bases = (i == 0) or (self.samples.n_ridge[mcmc_use[i]] != self.samples.n_ridge[mcmc_use[i-1]])
            if self.samples.n_ridge[mcmc_use[i]] > 0:
                basis_idx = slice(0, 1)
                for j in range(self.samples.n_ridge[mcmc_use[i]]):
                    if self.samples.ridge_type[mcmc_use[i]][j] == "cont":
                        basis_idx = slice(basis_idx.stop, basis_idx.stop + self.prior.df_spline)
                        n_act = self.samples.n_act[mcmc_use[i]][j]
                        knots = self.samples.knots[mcmc_use[i]][j].copy()
                        if (calc_all_bases or
                            n_act != self.samples.n_act[mcmc_use[i-1]][j] or
                            knots[0] != self.samples.knots[mcmc_use[i-1]][j][0]):
                                feat = self.samples.feat[mcmc_use[i]][j][:n_act].copy()
                                proj_dir = self.samples.proj_dir[mcmc_use[i]][j][:n_act].copy()
                                proj = newdata_s[:, feat] @ proj_dir
                                ridge_basis[j] = get_mns_basis(proj, knots) # Get basis function
                    else: # No continuous features in this basis
                        basis_idx = slice(basis_idx.stop, basis_idx.stop + 1)
                        n_act = self.samples.n_act[mcmc_use[i]][j]
                        if self.samples.ridge_type[j] == "cat": # all categorical features in this basis
                            if calc_all_bases:
                                feat = self.samples.feat[mcmc_use[i]][j][:n_act].copy()
                                ridge_basis[j] = get_cat_basis(newdata_s[:, feat])
                        else:  # some discrete quantitative features in this basis
                            proj_dir = self.samples.proj_dir[mcmc_use[i]][j][:n_act].copy()
                            if (calc_all_bases or
                                n_act != self.samples.n_act[mcmc_use[i-1]][j] or
                                np.any(proj_dir != self.samples.proj_dir[mcmc_use[i-1]][j][:n_act])):
                                    feat = self.samples.feat[mcmc_use[i]][j][:n_act].copy()
                                    ridge_basis[j] = newdata_s[:, feat] @ proj_dir
                    # Add predictions for jth basis function
                    preds[i] += ridge_basis[j] @ self.samples.coefs[mcmc_use[i], basis_idx]

        return preds
    
    def sobol(self, mcmc_use="last", n_mc=2**12):
        if mcmc_use == "last":
            mcmc_use = [self.specs.n_keep - 1]
        elif mcmc_use == "all":
            mcmc_use = list(range(self.specs.n_keep))
        elif type(mcmc_use) is np.ndarray:
            mcmc_use = list(mcmc_use.astype(int))
        elif type(mcmc_use) is int:
            mcmc_use = [mcmc_use]

        # Generate random samples of parameters according to Saltelli (2010) method.
        qrng = stats.qmc.Sobol(d=2 * self.data.p, scramble=True)
        base_sequence = qrng.random(n_mc)
        saltelli_sequence = np.zeros([(self.data.p + 2) * n_mc, self.data.p])

        idx = 0
        for i in range(n_mc):
            # Copy matrix "A"
            for j in range(self.data.p):
                saltelli_sequence[idx, j] = base_sequence[i, j]
            idx += 1

            # Cross-sample elements of "B" into "A"
            for k in range(self.data.p):
                for j in range(self.data.p):
                    if (j == k):
                        saltelli_sequence[idx, j] = base_sequence[i, j + self.data.p]
                    else:
                        saltelli_sequence[idx, j] = base_sequence[i, j]
                idx += 1

            # Copy matrix "B"
            for j in range(self.data.p):
                saltelli_sequence[idx, j] = base_sequence[i, j + self.data.p]
            idx += 1

        xmin = np.min(self.data.X, axis=0)
        xrange = np.max(self.data.X, axis=0) - xmin
        saltelli_sequence *= xrange
        saltelli_sequence += xmin
        NY = saltelli_sequence.shape[0]

        # Evaluate model at those param values
        mod_at_all_params = self.predict(saltelli_sequence, 
                                         mcmc_use=np.array(mcmc_use))
        del saltelli_sequence
    
        step = 2 + self.data.p
        mod_at_A = mod_at_all_params[:, 0:NY:step].copy()
        mod_at_B = mod_at_all_params[:, (step-1):NY:step].copy()
        mod_at_AB = [
            mod_at_all_params[:, (j + 1):NY:step].copy()
            for j in range(self.data.p)
        ]

        del mod_at_all_params

        first_order = np.zeros((len(mcmc_use), self.data.p))
        total_order = np.zeros((len(mcmc_use), self.data.p))
        
        for j in range(self.data.p):
            first_order[:, j] = np.mean(
                mod_at_B * (mod_at_AB[j] - mod_at_A), axis=1
            )

            total_order[:, j] = 0.5 * np.mean(
                (mod_at_A - mod_at_AB[j]) ** 2, axis=1
            )

        # Get all mcmc samples
        self.first_order_sobol = first_order
        self.total_order_sobol = total_order

        return
    
    def plot_sobol(self, labels=None, file=None):
        if labels is None:
            labels = ['v' + str(j+1) for j in range(self.data.p)]
        
        lightblue = (0.55, 0.65, 0.8)
        darkgrey = (0.15, 0.15, 0.15)
        
        # First-order Sobol'
        first_order = np.mean(self.first_order_sobol, axis=0)
        normalized_first_order = first_order / self.first_order_sobol.sum()
        
        # Total-order Sobol'
        total_order = np.mean(self.total_order_sobol, axis=0)
        
        # Creating first-order barplot
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.set_xlabel('Input Variable')
        ax1.set_ylabel("Sobol' Index")
        ax1.set_title("First-Order Sobol' Index")
        ax1.bar(labels, first_order, color=darkgrey)

        # Instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()  
        ax2.set_ylabel("Normalized Sobol' Index")
        ax2.plot(labels, normalized_first_order, marker='', linestyle='None')
        ax2.set_ylim(0, normalized_first_order.max()*1.05)
        
        # Creating total-order barplot        
        ax3.set_xlabel('Input Variable')
        ax3.set_ylabel("Total Sobol'")
        ax3.set_title("Total-Order Sobol' Index")
        ax3.bar(labels, total_order, color=lightblue, label='Higher-Order')
        ax3.bar(labels, first_order, color=darkgrey, label='First-Order')
        ax3.legend()        
    
        plt.tight_layout()
    
        if file is not None:
            plt.savefig(file)
        else:
            plt.show()

        plt.close(fig)
        
        return
    
    
    def plot(self, X_test=None, y_test=None, n_plot=None, coverage_target=0.95, file=None):
        # Get X and y
        if (X_test is None and y_test is not None) or (X_test is not None and y_test is None):
            raise ValueError("Both X_test and y_test should be specified or left as None.")
        
        if X_test is None:
            X = self.data.X.copy()
        else:
            X = X_test.copy()
            
        if y_test is None:
            y = self.data.y.copy()
        else:
            y = y_test.copy()
            
        n = len(y)
        if n_plot is None:
            n_plot = min(n, 1000)
        elif n_plot == 'all':
            n_plot = n
        elif n_plot > n:
            n_plot = n
        
        if n_plot < n:
            idx_plot = np.random.choice(n, n_plot, replace=False)
        else:
            idx_plot = np.arange(n)
        
        # Get predictions
        mn_samples = self.predict(X)
        post_mn = np.mean(mn_samples, axis=0)
        resid = y - post_mn
        bias = np.mean(resid)
        rmse = np.std(resid)
        R_squared = 1 - rmse**2 / np.var(y)
        
        # Get uq
        sd_samples = np.repeat(np.sqrt(self.samples.s2), n).reshape(mn_samples.shape)
        y_samples = np.random.normal(mn_samples, sd_samples)
        q_lower = (1.0 - coverage_target)/2.0
        q_upper = (1.0 + coverage_target)/2.0
        post_lower = np.quantile(y_samples, q_lower, axis=0)
        post_upper = np.quantile(y_samples, q_upper, axis=0)   
        coverage_est = np.mean(np.logical_and(
            y >= post_lower,
            y <= post_upper
            ))
        
        # make plots
        fig = plt.figure(figsize=(8, 6), dpi=100.0)
        lightblue = (0.55, 0.65, 0.8)
        darkgrey = (0.15, 0.15, 0.15)
        
        # Predicted v. actual
        fig.add_subplot(2, 2, 1)
        plt.scatter(
            y[idx_plot], post_mn[idx_plot],
            color=lightblue,
            s=15,
            alpha=0.5
            )
        plt.plot((min(y), max(y)), (min(post_mn), max(post_mn)),
                 color = darkgrey)
        plt.xlabel('Actual Response')
        plt.ylabel('Predicted Response')
        plt.title(f'Accuracy: RMSE = {rmse:.{3}g}, $R^2 = ${R_squared:.{3}f}')
        
        # Prediction Intervals
        fig.add_subplot(2, 2, 2)
        idx_sort = np.argsort(post_mn[idx_plot])
        plt.errorbar(list(range(n_plot)), post_mn[idx_plot][idx_sort],
                     [post_mn[idx_plot][idx_sort] - post_lower[idx_plot][idx_sort],
                      post_upper[idx_plot][idx_sort] - post_mn[idx_plot][idx_sort]],
                     fmt='none', color=lightblue,
                     label="Uncertainty Bound",
                     zorder=1)
        plt.scatter(list(range(n_plot)), y[idx_plot][idx_sort],
                    s=5, color='firebrick',
                    label = 'Actual Response', zorder=2, alpha=0.5)
        plt.plot(list(range(n_plot)), post_mn[idx_plot][idx_sort],
                    color=darkgrey, label = 'Predicted Response', zorder=3)
        plt.xlabel('Index')
        plt.ylabel('Response')
        plt.title(f'{100*coverage_target:.{1}f}% Intervals: Coverage = {100*coverage_est:.{1}f}%')
        plt.legend()
                
        # Residuals v. Predicted
        fig.add_subplot(2, 2, 3)
        plt.scatter(
            post_mn[idx_plot], resid[idx_plot],
            color=lightblue,
            s=15,
            alpha=0.5
            )
        plt.axhline(y = 0, color = darkgrey)
        plt.xlabel('Predicted Response')
        plt.ylabel('Residual')
        plt.title('Equal Variance and Lack of Trend')
        
        # Histogram of Residuals
        fig.add_subplot(2, 2, 4)
        xx = np.linspace(min(resid), max(resid), 100)
        norm_pdf_xx = stats.norm.pdf(xx, bias, rmse)
        plt.hist(
            resid,
            color = lightblue,
            edgecolor=darkgrey,
            density=True,
            bins=25,
            alpha=0.85
            )
        plt.plot(xx, norm_pdf_xx,
                 linewidth=2,
                 color=darkgrey
                 )
        plt.xlabel('Residual')
        plt.ylabel('Density')
        plt.title('Normality')
        
        fig.tight_layout()

        if file is not None:
            plt.savefig(file)
        else:
            plt.show()

        plt.close(fig)
        
        return
    
    def traceplot(self, file=None):
        # make plots
        fig = plt.figure(figsize=(8, 6), dpi=100.0)
        lightblue = (0.55, 0.65, 0.8)
        darkgrey = (0.15, 0.15, 0.15)
        
        # Number of ridge functions
        fig.add_subplot(2, 2, 1)
        plt.plot(
            self.samples.n_ridge,
            color=lightblue,
            label = 'Samples'
            )
        plt.axhline(y = np.mean(self.samples.n_ridge),
                    color = darkgrey, label = 'Mean')
        plt.xlabel('MCMC Iteration')
        plt.ylabel('Number of Ridge Functions')
        ess = effective_sample_size(self.samples.n_ridge)
        subchains = split_chain_into_subchains(self.samples.n_ridge, 4)
        if (np.var(subchains, axis=1) > 0).all():
            rhat = calculate_rhat(subchains)
            plt.title(f'n_ridge: ESS = {ess:.{0}f}, $\hat{{R}}$ = {rhat:.{3}f}')
        else:
            plt.title(f'n_ridge: ESS = {ess:.{0}f}, $\hat{{R}}$ = NA')
        plt.legend()
        
        # Residual Variance
        fig.add_subplot(2, 2, 2)
        plt.plot(
            self.samples.s2,
            color=lightblue,
            label = 'Samples'
            )
        plt.axhline(y = np.mean(self.samples.s2),
                    color = darkgrey, label = 'Mean')
        plt.xlabel('MCMC Iteration')
        plt.ylabel('Residual Variance')
        ess = effective_sample_size(self.samples.s2)
        subchains = split_chain_into_subchains(self.samples.s2, 4)
        rhat = calculate_rhat(subchains)
        plt.title(f's2: ESS = {ess:.{0}f}, $\hat{{R}}$ = {rhat:.{3}f}')
        plt.legend()
                
        # Coefficient Variance
        fig.add_subplot(2, 2, 3)
        ess = effective_sample_size(self.samples.var_coefs)
        subchains = split_chain_into_subchains(self.samples.var_coefs, 4)
        rhat = calculate_rhat(subchains)
        plt.plot(
            self.samples.var_coefs,
            color=lightblue,
            label = 'Samples'
            )
        plt.axhline(y = np.mean(self.samples.var_coefs),
                    color = darkgrey, label = 'Mean')
        plt.xlabel('MCMC Iteration')
        plt.ylabel('Variance of Basis Coefficients')
        ess = effective_sample_size(self.samples.var_coefs)
        subchains = split_chain_into_subchains(self.samples.var_coefs, 4)
        rhat = calculate_rhat(subchains)
        plt.title(f'var_coefs: ESS = {ess:.{0}f}, $\hat{{R}}$ = {rhat:.{3}f}')
        plt.legend()
        
        fig.tight_layout()

        if file is not None:
            plt.savefig(file)
        else:
            plt.show()

        plt.close(fig)
        
        return


def bppr(X, y, n_ridge_mean=10.0, n_ridge_max=None, n_act_max=None,
         df_spline=4, prob_relu=2/3, prior_coefs="zs", shape_var_coefs=None,
         scale_var_coefs=None, n_dat_min=None, scale_proj_dir_prop=None,
         adapt_act_feat=True, w_n_act=None, w_feat=None, n_post=1000,
         n_burn=9000, n_adapt=0, n_thin=1, silent=False):
    
    t0 = time.time()

    
    # Organize input into data, prior, and specs
    data = bpprData(X, y)
    prior = bpprPrior(n_ridge_mean, n_ridge_max, n_act_max, df_spline,
                      prob_relu, prior_coefs, shape_var_coefs, scale_var_coefs,
                      n_dat_min)
    specs = bpprSpecs(n_post, n_burn, n_adapt, n_thin, w_n_act, w_feat,
                      adapt_act_feat, scale_proj_dir_prop)
    
    # Pre-processing
    data.summarize(prior)
    prior.calibrate(data)
    specs.calibrate(data, prior)
    
    # Initialize the state of the Markov Chain
    state = bpprState(data, prior, specs)
    
    # Initialize the posterior samples
    samples = bpprSamples(prior, specs, state)
        
    # Run MCMC
    if specs.n_draws > 1:
        for it in range(1, specs.n_draws):
            if it == (specs.n_adapt):
                if specs.n_burn > 0:
                    state.phase = 'burn'
                else:
                    state.phase = 'post-burn'
                    
            if it == (specs.n_pre):
                state.phase = 'post-burn'
                
            # Update the state
            state.update(data, prior, specs)
            
            if state.phase == 'post-burn':
                # Write to samples
                samples.writeState(state)
                state.idx += 1
                
            if not silent and it % 500 == 0:
                print('\rBayesPPR MCMC {:.1%} Complete'.format(it / specs.n_draws), end='')

    t1 = time.time()
    if not silent:
        print('\rBayesPPR MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))    
    
    model = bpprModel(data, prior, specs, samples)
    
    return model

