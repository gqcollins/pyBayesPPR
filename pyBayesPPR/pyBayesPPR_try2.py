"""
Author: Gavin Collins
"""

import numpy as np
import scipy as sp
from itertools import combinations, chain
from scipy.special import comb, betaln
import time
import copy
from warnings import warn

X = np.random.normal(size = (100, 3))
y = X @ np.random.normal(size = 3) + np.random.normal(size = 100)
n_ridge_mean = 10; n_ridge_max = None; n_act_max = None; df_spline = 4; prob_relu = 2/3; prior_coefs = "zs"; shape_var_coefs = None; scale_var_coefs = None; n_dat_min = None; scale_proj_dir_prop = None; adapt_act_feat = True; w_n_act = None; w_feat = None; n_post = 1000; n_burn = 9000; n_adapt = 0; n_thin = 1; print_every = 1000; bppr_init = None


def get_qf_info(BtB, Bty):
    """Get the quadratic form y'B solve(B'B) B'y, as well as least squares coefs and cholesky of B'B"""
    try:
        chol_BtB = sp.linalg.cholesky(BtB, lower=False)  # might be a better way to do this with sp.linalg.cho_factor
    except np.linalg.LinAlgError as e:
        return None
    d = np.diag(chol_BtB)
    if len(d) > 1:
        if np.max(d[1:]) / np.min(d) > 1e3:
            return None
        
    bhat = sp.linalg.solve_triangular(chol_BtB, sp.linalg.solve_triangular(chol_BtB, Bty, trans=1))
    qf = Bty.T @ bhat
    return {'chol': chol_BtB, 'ls_est': bhat, 'qf': qf}

def append_qf_inv_chol(qf_info, dim):
  qf_info['inv_chol'] = sp.linalg.solve_triangular(qf_info['chol'], np.identity(dim))
  return qf_info

def print_update(it, n_draws, phase, current_time, start_time, n_ridge):
    print('MCMC iteration ' + str(it) + '/' + str(n_draws) + ' (' + phase + ') -- ' + str(round(current_time - start_time)) + ' secs -- n_ridge: ' + str(n_ridge) + '\n')

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

def get_cat_basis(X):
    p = np.shape(X)[1]
    if p > 1:
        tp = (1 - X[:, 0])
        for j in range(1,p):
            tp *= (1 - X[:, j])
        basis = 1 - tp
    else:
        basis = X
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
    for k in range(1, df + 1):
      d.append((r[k] - r[df]) / (knots[df + 1] - knots[k]))
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


def bppr(X, y, n_ridge_mean=10.0, n_ridge_max=None, n_act_max=None, df_spline=4, prob_relu=2/3, prior_coefs="zs", shape_var_coefs=None, scale_var_coefs=None, n_dat_min=None, scale_proj_dir_prop=None, adapt_act_feat=True, w_n_act=None, w_feat=None, n_post=1000, n_burn=9000, n_adapt=0, n_thin=1, print_every=1000, bppr_init=None):
    # Manage posterior draws
    assert n_thin <= n_post, 'n_thin > n_post. No posterior samples will be obtained.'
    n_post -= n_post % n_thin
    n_keep = int(np.floor(n_post/n_thin))
    n_pre = n_adapt + n_burn
    n_draws = n_pre + n_post
    idx = np.append(
        np.zeros(n_pre, dtype=int),
        np.repeat(np.arange(n_keep), n_thin)
        )
    if prior_coefs == 'flat':
        n_adapt = n_pre
        n_burn = 0

    # Pre-processing
    n = len(y)
    p = np.shape(X)[1]

    if w_feat is None:
        w_feat = np.ones(p)
    w_feat_norm = w_feat/np.sum(w_feat)

    mn_X = np.zeros(p)
    sd_X = np.ones(p)
    X_st = X.copy()
    feat_type = []
    for j in range(p):
        n_unique = len(set(X[:, j]))
        if n_unique <= 2:
            if n_unique == 1:
                feat_type.append('')
                w_feat[j] = 0.0
                w_feat_norm = w_feat/np.sum(w_feat)
            else:
                feat_type.append('cat')
        else:
            mn_X[j] = np.mean(X[:, j])
            sd_X[j] = np.std(X[:, j])
            X_st[:, j] = (X[:, j] - mn_X[j]) / sd_X[j]
            if n_unique <= df_spline:
                feat_type.append('disc')
            else:
                feat_type.append('cont')
    feat_type = np.array(feat_type)

    if n_act_max is None:
        n_cat = np.sum(feat_type == np.repeat('cat', p))
        n_act_max = int(min(3, p - n_cat) + min(3, np.ceil(n_cat/2)))

    if w_n_act is None:
        w_n_act = np.ones(n_act_max)
    w_n_act_norm = w_n_act/np.sum(w_n_act)

    if scale_proj_dir_prop is None:
        proj_dir_prop_prec = 1000.0  # scale_proj_dir_prop = 0.002
    else:
        assert scale_proj_dir_prop > 0 and scale_proj_dir_prop <= 1, "scale_proj_dir_prop must be in (0, 1]"
        inv_scale_proj_dir_prop = 1/scale_proj_dir_prop
        proj_dir_prop_prec = (inv_scale_proj_dir_prop - 1) + np.sqrt(inv_scale_proj_dir_prop * (inv_scale_proj_dir_prop - 1))

    if n_dat_min is None:
        n_dat_min = min(20, 0.1 * n)

    if n_dat_min <= df_spline:
        warn('n_dat_min too small. If n_dat_min was set by default, df_spline is large compared to the sample size. Setting n_dat_min = df_spline + 1')
        n_dat_min = df_spline + 1

    # Maximum proportion of inactive datapoints in each ridge function
    p_dat_max = 1 - n_dat_min / n

    if n_ridge_max is None:
        n_ridge_max = int(min(150, np.floor(n/df_spline) - 2))
    assert n_ridge_max > 0, 'n_ridge_max <= 0. If n_ridge_max was set by default, df_spline is too large compared to the sample size.'

    # Quantiles for knot locations (except initial knot)
    knot_quants = np.linspace(0.0, 1.0, num=df_spline + 1)

    assert prior_coefs in ['zs', 'flat'], "prior_coefs must be either 'zs' or 'flat'"
    if prior_coefs == 'zs':
      if shape_var_coefs is None:
          shape_var_coefs = 0.5
      if scale_var_coefs is None:
          scale_var_coefs = n/2
      var_coefs = np.zeros(n_keep)
    elif prior_coefs == 'flat':
        var_coefs = None
        c_var_coefs = 1.0

    sd_resid = np.zeros(n_keep) # Error standard deviation
    coefs = np.zeros([n_keep, df_spline * n_ridge_max + 1]) # Basis coefficients
    n_ridge = np.zeros(n_keep, dtype=int)  # Number of ridge functions
    n_act = np.zeros([n_keep, n_ridge_max], dtype=int) # Number of active features for jth ridge function
    feat = np.zeros([n_keep, n_ridge_max, n_act_max], dtype=int)  # Features being used in jth ridge function
    knots = np.zeros([n_keep, n_ridge_max, df_spline + 2]) # Location of knots for nsplines
    proj_dir = np.zeros([n_keep, n_ridge_max, n_act_max])  # Ridge directions
    
    # prior mean for proj_dir (arbitrary, since prior precision is zero)
    proj_dir_mn = [np.repeat(1/np.sqrt(a), a) for a in range(1,n_act_max+1)]

    # Number of basis functions in each ridge function (first ridge function is the intercept)
    n_basis_ridge = [1]
    ridge_type = []
    j_quant = []  # which ridge functions are quantitative?
    n_quant = 0  # how many ridge functions are quantitative?
    basis_mat = np.ones((n,1))  # Current basis matrix
    basis_idx = [slice(0)]  # Current indices of segments of basis functions
    BtB = np.zeros((n_ridge_max * df_spline + 1, n_ridge_max * df_spline + 1))
    Bty = np.zeros(n_ridge_max * df_spline + 1)

    # Initialization
    if bppr_init is None:
        if prior_coefs == 'zs':
            var_coefs[0] = scale_var_coefs / shape_var_coefs
            c_var_coefs = var_coefs[0] / (var_coefs[0] + 1)
        sd_resid[0] = 1.0
        coefs[0, 0] = np.mean(y)
    else:
        assert bppr_init is None, "initialization not supported yet"
    # }else{
    #     if(prior_coefs == 'zs'){
    #         var_coefs[1] = bppr_init$var_coefs
    #         c_var_coefs = var_coefs[1] / (var_coefs[1] + 1)
    #     }
    #     sd_resid[1] = bppr_init$sd_resid
    #     coefs[[1]] = bppr_init$coefs
    #     n_ridge[1] = bppr_init$n_ridge
    #     n_act[[1]] = bppr_init$n_act
    #     feat[[1]] = bppr_init$feat
    #     knots[[1]] = bppr_init$knots
    #     proj_dir[[1]] = bppr_init$proj_dir
    #     if(n_ridge[1] > 0){
    #         basis_idx_start = 2
    #         for(j in 1: n_ridge[1]){
    #             if(any(feat_type[feat[[1]][[j]]] == 'cont')){
    #                 ridge_type[j] = 'cont'
    #                 n_basis_ridge[j + 1] = df_spline
    #                 j_quant = c(j_quant, j)
    #                 proj = X_st[, feat[[1]][[j]], drop = FALSE] % * % proj_dir[[1]][[j]]
    #                 basis_mat = cbind(basis_mat, get_mns_basis(
    #                     proj, knots[[1]][[j]]))  # Get basis function
    #                 basis_idx[[j + 1]] = basis_idx_start: (basis_idx_start + df_spline - 1)
    #                 basis_idx_start = basis_idx_start + df_spline
    #             }else{
    #                 n_basis_ridge[j + 1] = 1
    #                 if(any(feat_type[feat[[1]][[j]]] == 'disc')){
    #                     ridge_type[j] = 'disc'
    #                     n_basis_ridge[j + 1] = 1
    #                     j_quant = c(j_quant, j)
    #                     basis_mat = cbind(basis_mat, X_st[, feat[[1]][[j]], drop=FALSE] % * % proj_dir[[1]][[j]])
    #                     basis_idx[[j + 1]] = basis_idx_start
    #                     basis_idx_start = basis_idx_start + 1
    #                 }else{
    #                     ridge_type[j] = 'cat'
    #                     n_basis_ridge[j] = 1
    #                     basis_mat = cbind(basis_mat, get_cat_basis(X_st[, feat[[1]][[j]], drop=FALSE]))
    #                     basis_idx[[j + 1]] = basis_idx_start
    #                     basis_idx_start = basis_idx_start + 1
    #                 }
    #             }
    #         }
    #         n_quant = length(j_quant)
    #     }
    #     w_n_act = bppr_init$w_n_act
    #     w_feat = bppr_init$w_feat
    # }

    n_basis_total = np.sum(n_basis_ridge)
    # inner product of basis_mat with itself
    BtB[:n_basis_total, :n_basis_total] = basis_mat.T @ basis_mat
    # inner product of basis_mat with y
    Bty[:n_basis_total] = basis_mat.T @ y
    qf_info = get_qf_info(BtB[:n_basis_total, :n_basis_total], Bty[:n_basis_total])
    if n_adapt == 0:
        qf_info = append_qf_inv_chol(qf_info, dim=n_basis_total)

    ssy = y.T @ y  # Keep track of overall sse
    sse = ssy - c_var_coefs * qf_info['qf']
    log_mh_bd = 0.0
    log_mh_act_feat = 0.0

    if n_adapt > 0:
        phase = 'adapt'
    elif n_burn > 0:
        phase = 'burn'
    else:
        phase = 'post-burn'
        
    if print_every > 0:
        start_time = time.time()
        print_update(1, n_draws, phase, time.time(), start_time, n_ridge[0])
        silent = False
    else:
        print_every = n_draws + 2
        silent = True
        
        
    # Run MCMC
    if n_draws > 1:
        for it in range(1,n_draws):
            # Set current it values to last it values (these will change during the iteration)
            if idx[it] > idx[it - 1]:
                if prior_coefs == 'zs':
                    var_coefs[idx[it]] = var_coefs[idx[it - 1]].copy()
                sd_resid[idx[it]] = sd_resid[idx[it - 1]].copy()
                coefs[idx[it]] = coefs[idx[it - 1]].copy()
                n_ridge[idx[it]] = n_ridge[idx[it - 1]].copy()
                n_act[idx[it]] = n_act[idx[it - 1]].copy()
                feat[idx[it]] = feat[idx[it - 1]].copy()
                knots[idx[it]] = knots[idx[it - 1]].copy()
                proj_dir[idx[it]] = proj_dir[idx[it - 1]].copy()

            if it == (n_adapt + 1):
                if n_burn > 0:
                    phase = 'burn'
                else:
                    phase = 'post-burn'
            if it == (n_pre + 1):
                phase = 'post-burn'
                
            if ((it % print_every) == 0)  or  ((it == n_adapt  or  it == n_pre) and not silent):
                print_update(it + 1, n_draws, phase, time.time(), start_time, n_ridge[idx[it]])

            move_type = get_move_type(n_ridge[idx[it]], n_quant, n_ridge_max)

            if move_type == 'birth':  # Birth step
                n_ridge_prop = n_ridge[idx[it]] + 1
                # Propose number of active features
                n_act_prop = np.random.choice(n_act_max, p=w_n_act_norm) + 1
                if adapt_act_feat:
                    # Nott, Kuk, Duc for n_act
                    log_mh_act_feat = -(np.log(n_act_max) + np.log(w_n_act_norm[n_act_prop-1]))
                    # Propose features to include
                    if n_act_prop == 1:
                        feat_prop = np.random.choice(p, 1)
                    else:
                        feat_prop = np.random.choice(p, n_act_prop, p=w_feat_norm, replace=False)
                    if n_act_prop > 1  and  n_act_prop < n_act_max:
                        log_mh_act_feat -= lchoose(p, n_act_prop) + np.log(dwallenius(w_feat_norm, feat_prop))  # Nott, Kuk, Duc for feat
                else:
                    # Propose features to include
                    feat_prop = np.random.choice(p, n_act_prop, p=w_feat_norm, replace = False)

                if np.all(feat_type[feat_prop] == 'cat'):  # Are all of the proposed features categorical?
                    ridge_type_prop = 'cat'
                    n_quant_prop = n_quant
                    proj_dir_prop = None
                    knots_prop = None
                    ridge_basis_prop = get_cat_basis(X_st[:, feat_prop])
                    n_basis_prop = 1
                else:
                    n_quant_prop = n_quant + 1
                    if n_act_prop == 1:
                        proj_dir_prop = np.random.choice([-1, 1], 1)
                    else:
                        # Propose direction
                        proj_dir_prop = rps(proj_dir_mn[n_act_prop - 1], 0.0)
                    proj_prop = X_st[:, feat_prop] @ proj_dir_prop # Get proposed projection
                    
                    if np.any(feat_type[feat_prop] == 'cont'):  # Are any proposed features continuous?
                        ridge_type_prop = 'cont'
                        max_knot0 = np.quantile(proj_prop, p_dat_max)
                        rg_knot0 = (max_knot0 - np.min(proj_prop)) / prob_relu
                        knot0_prop = max_knot0 - rg_knot0 * np.random.uniform()
                        knots_prop = np.append([knot0_prop], np.quantile(proj_prop[proj_prop > knot0_prop], knot_quants))  # Get proposed knots
                        # Get proposed basis functions
                        ridge_basis_prop = get_mns_basis(proj_prop, knots_prop)
                        n_basis_prop = df_spline
                        # The proposed features are a mix of categorical and discrete quantitative
                    else:
                        ridge_type_prop = 'disc'
                        knots_prop = None
                        ridge_basis_prop = proj_prop
                        n_basis_prop = 1

                # inner product of proposed new basis functions
                PtP = ridge_basis_prop.T @ ridge_basis_prop
                BtP = basis_mat.T @ ridge_basis_prop
                Pty = ridge_basis_prop.T @ y

                j_birth = n_ridge[idx[it]]
                basis_idx_start = basis_idx[j_birth].stop + 1
                basis_idx_prop = slice(basis_idx_start, basis_idx_start + n_basis_prop)
                BtB[:n_basis_total, basis_idx_prop] = BtP
                BtB[basis_idx_prop, :n_basis_total] = BtP.T
                BtB[basis_idx_prop, basis_idx_prop] = PtP
                Bty[basis_idx_prop] = Pty

                qf_info_prop = get_qf_info(BtB[:(n_basis_total + n_basis_prop), :(n_basis_total + n_basis_prop)],
                                           Bty[:(n_basis_total + n_basis_prop)])
                log_mh_bd_prop = get_log_mh_bd(n_ridge_prop, n_quant_prop, n_ridge_max)

                if qf_info_prop is not None:
                    if qf_info_prop['qf'] < ssy:
                        sse_prop = ssy - c_var_coefs * qf_info_prop['qf']

                        # Compute the acceptance probability
                        log_mh = (
                            log_mh_bd - log_mh_bd_prop + log_mh_act_feat + # Adjustment for probability of birth proposal
                            -n/2 * (np.log(sse_prop) - np.log(sse)) + # Part of the marginal likelihood
                            np.log(n_ridge_mean/(n_ridge[idx[it]] + 1)) # Prior and proposal distribution
                            )
                        if prior_coefs == 'zs':
                            # The rest of the marginal likelihood for Zellner-Siow prior
                            log_mh -= n_basis_prop * np.log(var_coefs[idx[it]] + 1)/2
                        else:
                            # The rest of the marginal likelihood for flat prior
                            log_mh += np.log(10e-6)

                        if np.log(np.random.uniform()) < log_mh:  # Accept the proposal
                            n_ridge[idx[it]] += 1
                            n_act[idx[it], j_birth] = n_act_prop
                            feat[idx[it], j_birth, :n_act_prop] = feat_prop
                            knots[idx[it], j_birth, :] = knots_prop
                            proj_dir[idx[it], j_birth, :n_act_prop] = proj_dir_prop

                            if ridge_type_prop != 'cat':
                                j_quant.append(j_birth)
                                n_quant = n_quant_prop
                            ridge_type.append(ridge_type_prop)

                            basis_idx.append(basis_idx_prop)
                            n_basis_ridge.append(n_basis_prop)
                            n_basis_total += n_basis_prop
                            basis_mat = np.append(basis_mat, ridge_basis_prop, axis=1)

                            # Update weights
                            if adapt_act_feat:
                                w_n_act[n_act_prop-1] += 1
                                w_n_act_norm = w_n_act/np.sum(w_n_act)
                                w_feat[feat_prop] += 1
                                w_feat_norm = w_feat/np.sum(w_feat)

                            qf_info = copy.deepcopy(qf_info_prop)
                            if it > n_adapt:
                                qf_info = append_qf_inv_chol(qf_info, dim=n_basis_total)
                            sse = sse_prop
                            log_mh_bd = log_mh_bd_prop

            elif move_type == 'death':  # Death step
                # Choose random index to delete
                j_death = np.random.choice(n_ridge[idx[it]])

                n_act_prop = n_act[idx[it]][j_death]
                feat_prop = feat[idx[it]][j_death, :n_act_prop]
                if adapt_act_feat:
                    log_mh_act_feat = np.log(n_act_max) + np.log((w_n_act[n_act_prop-1] - 1)/(np.sum(w_n_act) - 1))
                    w_feat_prop = w_feat.copy()
                    w_feat_prop[feat_prop] -= 1
                    w_feat_prop_norm = w_feat_prop/np.sum(w_feat_prop)
                    if n_act_prop > 1:
                        log_mh_act_feat += lchoose(p, n_act_prop) + np.log(dwallenius(w_feat_prop_norm, feat_prop))  # Nott, Kuk, and Duc

                idx_prop = list(range(n_basis_total))
                del idx_prop[basis_idx[j_death + 1]]             
                qf_info_prop = get_qf_info(BtB[np.ix_(idx_prop, idx_prop)],
                                           Bty[idx_prop])
                
                n_ridge_prop = n_ridge[idx[it]] - 1
                if ridge_type[j_death] == 'cat':
                    n_quant_prop = n_quant
                else:
                    n_quant_prop = n_quant - 1

                log_mh_bd_prop = get_log_mh_bd(n_ridge_prop, n_quant_prop, n_ridge_max)

                if qf_info_prop is not None:
                    if qf_info_prop['qf'] < ssy:
                        sse_prop = ssy - c_var_coefs * qf_info_prop['qf']
                        n_basis_prop = n_basis_ridge[j_death + 1]

                        # Compute acceptance probability
                        log_mh = (log_mh_bd - log_mh_bd_prop + log_mh_act_feat + # Adjustment for probability of death proposal
                                  -n/2 * (np.log(sse_prop) - np.log(sse)) + # Part of the marginal likelihood
                                  np.log(n_ridge[idx[it]]/n_ridge_mean) # Prior and proposal distribution
                                  )
                        
                        if prior_coefs == 'zs':
                            # The rest of the marginal likelihood for Zellner-Siow prior
                            log_mh += 0.5 * n_basis_prop * np.log(var_coefs[idx[it]] + 1)
                        else:
                            # The rest of the marginal likelihood for flat prior
                            log_mh = log_mh - np.log(10e-6)

                        if np.log(np.random.uniform()) < log_mh:  # Accept the proposal
                            n_basis_total -= n_basis_prop
                            n_ridge[idx[it]] -= 1
                            if ridge_type[j_death] != 'cat':
                                n_quant -= 1
                                j_quant.remove(j_death)

                            for k in range(len(j_quant)):
                                if j_quant[k] > j_death:
                                    j_quant[k] -= 1
                                
                            del ridge_type[j_death]
                            
                            basis_mat = np.delete(basis_mat, basis_idx[j_death+1], axis=1)
                            BtB[:n_basis_total, :n_basis_total] = BtB[np.ix_(idx_prop, idx_prop)]
                            Bty[:n_basis_total] = Bty[idx_prop]
                            for j in range(j_death, n_ridge[idx[it]]):
                                basis_idx[j + 1] = slice(basis_idx[j + 1].start - n_basis_ridge[j_death + 1],
                                                         basis_idx[j + 1].stop - n_basis_ridge[j_death + 1]
                                                         )
                            del basis_idx[j_death + 1]
                            del n_basis_ridge[j_death + 1]

                            # Update weights
                            if adapt_act_feat:
                                w_feat = w_feat_prop.copy()
                                w_feat_norm = w_feat/np.sum(w_feat)
                                w_n_act[n_act_prop-1] -= 1
                                w_n_act_norm = w_n_act/np.sum(w_n_act)

                            qf_info = qf_info_prop.copy()
                            if it > n_adapt:
                                qf_info = append_qf_inv_chol(qf_info, dim=n_basis_total)
                            sse = sse_prop
                            log_mh_bd = log_mh_bd_prop

            else:  # Change Step
                j_change = np.random.choice(j_quant) # Which ridge function should we change?
                
                n_act_prop = n_act[idx[it]][j_change].copy()
                feat_prop = feat[idx[it]][j_change][:n_act_prop].copy()
                proj_dir_curr = proj_dir[idx[it]][j_change][:n_act_prop].copy()
                
                # Get proposed direction
                if n_act[idx[it]][j_change] == 1:
                    proj_dir_prop = np.random.choice([-1, 1], 1)
                else:
                    proj_dir_prop = rps(proj_dir_curr, proj_dir_prop_prec)

                proj_prop = X_st[:,feat_prop] @ proj_dir_prop # Get proposed projection

                if ridge_type[j_change] == 'cont':  # Are any variables continuous for this ridge function?
                    max_knot0 = np.quantile(proj_prop, p_dat_max)
                    rg_knot0 = (max_knot0 - np.min(proj_prop)) / prob_relu
                    knot0_prop = max_knot0 - rg_knot0 * np.random.uniform()
                    knots_prop = np.append([knot0_prop], np.quantile(proj_prop[proj_prop > knot0_prop], knot_quants))  # Get proposed knots
                    ridge_basis_prop = get_mns_basis(proj_prop, knots_prop) # Get proposed basis function
                else:
                    knots_prop = None
                    ridge_basis_prop = proj_prop

                # inner product of proposed new basis functions
                PtP = ridge_basis_prop.T @ ridge_basis_prop
                BtP = basis_mat.T @ ridge_basis_prop
                Pty = ridge_basis_prop.T @ y

                BtB_prop = BtB[:n_basis_total, :n_basis_total].copy()
                BtB_prop[basis_idx[j_change + 1], :] = BtP.T.copy()
                BtB_prop[:, basis_idx[j_change + 1]] = BtP.copy()
                BtB_prop[basis_idx[j_change + 1], basis_idx[j_change + 1]] = PtP.copy()

                Bty_prop = Bty[:n_basis_total].copy()
                Bty_prop[basis_idx[j_change + 1]] = Pty.copy()

                qf_info_prop = get_qf_info(BtB_prop, Bty_prop)

                if qf_info_prop is not None:
                    if qf_info_prop['qf'] < ssy:
                        sse_prop = ssy - c_var_coefs * qf_info_prop['qf']

                        # Compute the acceptance probability
                        log_mh = -n/2 * (np.log(sse_prop) - np.log(sse))  # Marginal Likelihood

                        if np.log(np.random.uniform()) < log_mh: # Accept the proposal
                            knots[idx[it], j_change, :] = knots_prop.copy()
                            proj_dir[idx[it], j_change, :n_act_prop] = proj_dir_prop.copy()

                            BtB[:n_basis_total, :n_basis_total] = BtB_prop.copy()
                            Bty[:n_basis_total] = Bty_prop.copy()
                            basis_mat[:, basis_idx[j_change + 1]] = ridge_basis_prop

                            qf_info = qf_info_prop.copy()
                            if it > n_adapt:
                                qf_info = append_qf_inv_chol(qf_info, dim=n_basis_total)
                            sse = sse_prop

            if it > n_adapt:
                if (it == n_adapt + 1)  and  ('inv_chol' not in qf_info):
                    qf_info = append_qf_inv_chol(qf_info, dim=n_basis_total)

                sd_resid[idx[it]] = np.sqrt(1/np.random.gamma(n/2, 2/sse))

                # Draw coefs
                coefs[idx[it]][:n_basis_total] = (c_var_coefs * qf_info['ls_est'] +
                                                  np.sqrt(c_var_coefs) * sd_resid[idx[it]] * qf_info['inv_chol'] @ np.random.normal(size = n_basis_total)
                                                  )

                if prior_coefs == 'zs':
                    qf_comp = qf_info['chol'] @ coefs[idx[it]][:n_basis_total]
                    qf2 = qf_comp.T @ qf_comp
                    
                    var_coefs[idx[it]] = 1/np.random.gamma(shape_var_coefs + n_basis_total/2,
                                                           1 / (scale_var_coefs + qf2/(2*sd_resid[idx[it]]**2))
                                                           )
                    c_var_coefs = var_coefs[idx[it]] / (var_coefs[idx[it]] + 1)
                    sse = ssy - c_var_coefs * qf_info['qf']

    if not silent:
        print_update(n_draws, n_draws, phase, time.time(), start_time, n_ridge[idx[it]])

    return (n_keep, n_ridge, n_act, feat, proj_dir, knots, coefs, sd_resid,
            var_coefs, mn_X, sd_X, X, y, n_ridge_mean, n_ridge_max, n_act_max, df_spline,
            prob_relu, prior_coefs, shape_var_coefs, scale_var_coefs, n_dat_min,
            scale_proj_dir_prop, adapt_act_feat, w_n_act, w_feat, n_post, n_burn, n_adapt,
            n_thin, print_every, bppr_init
            )

# X = np.random.normal(size = (100, 3))
# y = np.random.normal(size = 100)
fit = bppr(X, y)
