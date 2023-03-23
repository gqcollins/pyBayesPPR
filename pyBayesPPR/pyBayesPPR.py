#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Gavin Collins
"""

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from itertools import combinations, chain
from scipy.special import comb
from collections import namedtuple
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
import time


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color='red')


relu = lambda a: (abs(a) + a) / 2 # same as max(0,a)


def makeBasis(proj, knots):
    """Make basis matrix using continuous covariates"""
    df = len(knots) - 2
    n = len(proj)
    basis = np.zeros((n, df))
    basis[:, 0] = relu(proj - knots[0])
    if df > 1:
        r = np.zeros((n, df + 1))
        d = np.zeros((n, df))
        for k in range(df + 1):
            r[:, k] = relu(proj - knots[k + 1])**3
            
        for k in range(df):
            d[:, k] = (r[:, k] - r[:, df]) / (knots[df + 1] - knots[k + 1])
            
        for k in range(df - 1):
            basis[:, k + 1] = d[:, k] - d[:, df - 1]
            
    return basis


def normalize(x, mean, sd):
    """Normalize to mean 0, sd 1"""
    return (x - mean) / sd


def combn_idx(n, k):
    """Get all combinations of indices from 0:n of length k"""
    # https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                        int, count=count * k)
                        
    return index.reshape(-1, k)


def dwallenius(wfeat_norm, feat):
    """Multivariate Walenius' noncentral hypergeometric density function with some variables fixed"""
    logMH = wfeat_norm[feat - 1] / sum(np.delete(wfeat_norm, feat))
    j = len(logMH)
    ss = 1 + (-1) ** j * 1 / (sum(logMH) + 1)
    for i in range(j - 1):
        idx = combn_idx(j, i + 1)
        temp = logMH[idx]
        ss = ss + (-1) ** (i + 1) * sum(1 / (temp.sum(axis=1) + 1))
        
    return ss


Qf = namedtuple('Qf', 'R bhat qf')

def getQf(BtB, Bty):
    """Get the quadratic form y'B solve(B'B) B'y, as well as least squares coefs and cholesky of B'B"""
    try:
        R = sp.linalg.cholesky(BtB, lower=False)  # might be a better way to do this with sp.linalg.cho_factor
    except np.linalg.LinAlgError as e:
        return None
    dr = np.diag(R)
    if len(dr) > 1:
        if max(dr[1:]) / min(dr) > 1e3:
            return None
        
    bhat = sp.linalg.solve_triangular(R, sp.linalg.solve_triangular(R, Bty, trans=1))
    qf = np.dot(bhat, Bty)
    return Qf(R, bhat, qf)


def getlogMHnActFeat(nAct, feat, wnAct_norm, wfeat_norm, p, nActMax):
    """Get (nAct,feat) term for RJMCMC MH acceptance ratio"""
    if nAct == 1:
        out = (np.log(wnAct_norm[nAct - 1]) - np.log(2 * p)  # proposal
               + np.log(2 * p) + np.log(nActMax))
               
    else:
        x = np.zeros(p)
        x[feat] = 1
        lprob_feat = np.log(dwallenius(wfeat_norm, feat))
        out = (np.log(wnAct_norm[nAct - 1]) + lprob_feat - nAct * np.log(2)  # proposal
               + nAct * np.log(2) + np.log(comb(p, nAct)) + np.log(nActMax))  # prior
    
    return out


def rps(nAct, mu, kappa): 
    """Generate random draw from the power-spherical distribution"""
    if nAct == 1:
        theta = np.random.choice((-1, 1), 1)
    else:
        uhat = -mu
        uhat[0] = uhat[0] + 1.0
        u = uhat / np.sqrt(np.sum(uhat**2))
        
        b = (nAct - 1.0)/2.0
        a = b + kappa
        z = np.random.beta(a, b)
        
        y = np.zeros(nAct)
        y[0] = 2.0*z - 1.0
        temp = np.random.normal(size = nAct - 1)
        v = temp / np.sqrt(np.sum(temp**2))
        y[1:] = np.sqrt(1.0 - y[0]**2) * v
        
        uy = np.dot(u, y)
        
        theta = y - 2.0 * u * uy
        
    return theta
    


CandidateRidge = namedtuple('CandidateRidge', 'basis nAct feat projDir knots logMHnActFeat')

def genCandRidge(nActMax, wnAct_norm, wfeat_norm, maxPropZero, probRelu, quantKnots, p, X):
    """Generate a candidate ridge function for birth step, as well as (nAct,feat) term for RJMCMC MH acceptance ratio"""
    nAct = int(np.random.choice(range(nActMax), p=wnAct_norm) + 1)
    if nAct == 1:
        feat = np.random.choice(p)
    else:
        feat = np.sort(np.random.choice(p, size=nAct, p=wfeat_norm, replace=False))
        
    projDir = rps(nAct, np.zeros(nAct), 0)
    proj = np.dot(X[:, feat].reshape(len(X), nAct), projDir)

    knot0max = np.quantile(proj, maxPropZero)
    knot0range = (knot0max - np.min(proj)) / probRelu
    knot0 = knot0max - knot0range * np.random.rand()
    knots = np.concatenate([np.array([knot0]), np.quantile(proj[proj > knot0], quantKnots)]) # Get proposed knots

    basis = makeBasis(proj, knots)
    logMHnActFeat = getlogMHnActFeat(nAct, feat, wnAct_norm, wfeat_norm, p, nActMax)
    return CandidateRidge(basis, nAct, feat, projDir, knots, logMHnActFeat)


RidgeChange = namedtuple('RidgeChange', 'basis projDir knots')

def genRidgeChange(projDir, precProjDirProp, nAct, feat, maxPropZero, probRelu, quantKnots, X):
    """Generate a condidate basis for change step"""
    projDirChange = rps(nAct, projDir, precProjDirProp)
    projChange = np.dot(X[:, feat].reshape(len(X), nAct), projDirChange)
    
    knot0maxChange = np.quantile(projChange, maxPropZero)
    knot0rangeChange = (knot0maxChange - np.min(projChange)) / probRelu
    knot0Change = knot0maxChange - knot0rangeChange * np.random.rand()
    knotsChange = np.concatenate([np.array([knot0Change]), np.quantile(projChange[projChange > knot0Change], quantKnots)]) # Get proposed knots

    basis = makeBasis(projChange, knotsChange)
    return RidgeChange(basis, projDirChange, knotsChange)


class bpprPrior:
    """Structure to store prior"""
    def __init__(self, nRidgeMean, nRidgeMax, nActMax, dfSpline, probRelu, shapeVarCoefs, scaleVarCoefs, minNonzero, precProjDirProp, wnAct, wfeat):
        self.nRidgeMean = nRidgeMean
        self.nRidgeMax = nRidgeMax
        self.nActMax = nActMax
        self.dfSpline = dfSpline
        self.probRelu = probRelu
        self.shapeVarCoefs = shapeVarCoefs
        self.scaleVarCoefs = scaleVarCoefs
        self.minNonzero = minNonzero
        self.precProjDirProp = precProjDirProp
        self.wnAct = wnAct
        self.wfeat = wfeat
        return


class bpprData:
    """Structure to store data"""
    def __init__(self, X, y):
        self.X_orig = X
        self.y = y
        self.ssy = sum(y * y)
        self.n = len(X)
        self.p = len(X[0])
        self.Xmn = np.zeros(self.p)
        self.Xsd = np.zeros(self.p)
        for i in range(self.p):
            self.Xmn[i] = np.mean(X[:, i])
            self.Xsd[i] = np.std(X[:, i])
        self.X = normalize(self.X_orig, self.Xmn, self.Xsd)
        return


Samples = namedtuple('Samples', 'sdResid varCoefs nRidge nRidge_models nBasis_models nAct feat projDir knots coefs')

class bpprState:
    """The current state of the RJMCMC chain, with methods for getting the log posterior and for updating the state"""
    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.sdResid = 1.
        self.varCoefs = 1.
        self.nRidge = 0
        self.R = 1
        self.wnAct = np.ones(prior.nActMax) * prior.wnAct
        self.wnAct_norm = self.wnAct / np.sum(self.wnAct)
        self.wfeat = np.ones(data.p) * prior.wfeat
        self.wfeat_norm = self.wfeat / np.sum(self.wfeat)
        self.basis = np.ones([data.n, 1])
        self.maxPropZero = 1 - prior.minNonzero / data.n
        self.nBasis = 1
        self.projDir = np.zeros([prior.nRidgeMax, prior.nActMax])
        self.quantKnots = np.linspace(0., 1., prior.dfSpline + 1) # All knots except knot0
        self.knots = np.zeros([prior.nRidgeMax, prior.dfSpline + 2]) # There are prior.dfSpline + 2 knots in total
        self.nAct = np.zeros([prior.nRidgeMax], dtype=int)
        self.feat = np.zeros([prior.nRidgeMax, prior.nActMax], dtype=int)
        self.Bty = np.zeros(prior.nRidgeMax * prior.dfSpline + 1)
        self.Bty[0] = np.sum(data.y)
        self.BtB = np.zeros([prior.nRidgeMax * prior.dfSpline + 1, prior.nRidgeMax * prior.dfSpline + 1])
        self.BtB[0, 0] = data.n
        self.R = np.array([[np.sqrt(data.n)]])  # np.linalg.cholesky(self.BtB[0, 0])
        self.R_inv_t = np.array([[1 / np.sqrt(data.n)]])
        self.bhat = np.mean(data.y)
        self.qf = pow(np.sqrt(data.n) * np.mean(data.y), 2)
        self.sse = data.ssy - 0.5 * self.qf
        self.logMHbd = 0 # np.log(1)
        self.count = np.zeros(3) # How many (birth, death, change) steps have been accepted?
        self.cmod = False  # has the state changed since the last write (i.e., has a birth, death, or change been accepted)?
        return


    def update(self):
        """Update the current state using a RJMCMC step (and Gibbs steps at the end of this function)"""

        if self.nRidge == 0:
            move_type = 1
            logMHbdCand = np.log(1/3)

        elif self.nRidge == self.prior.nRidgeMax:
            move_type = np.random.choice(np.array([2, 3]))
            if move_type == 2: # Death step
                logMHbdCand = np.log(1/3)
                
            else: # Change step
                logMHbdCand = np.log(2/3)

        else:
            move_type = np.random.choice([1, 2, 3])
            if self.nRidge == (self.prior.nRidgeMax - 1)  &  move_type == 1: # Birth step
                logMHbdCand = np.log(2/3)
                
            elif self.nRidge == 1  &  move_type == 2: # Death step
                logMHbdCand = 0 # np.log(1)
                
            else:
                logMHbdCand = np.log(1/3)
            
            
        if move_type == 1:
            ## BIRTH step

            cand = genCandRidge(self.prior.nActMax, self.wnAct_norm, self.wfeat_norm, self.maxPropZero, self.prior.probRelu, self.quantKnots, self.data.p, self.data.X)

            CtC = np.dot(cand.basis.T, cand.basis)
            BtC = np.dot(self.basis.T, cand.basis)
            Cty = np.dot(cand.basis.T, self.data.y)
            
            Cind = slice(self.nBasis, self.nBasis + self.prior.dfSpline)
            self.Bty[Cind] = Cty
            self.BtB[0:self.nBasis, Cind] = BtC
            self.BtB[Cind, 0:(self.nBasis)] = BtC.T
            self.BtB[Cind, Cind] = CtC

            qf_cand = getQf(self.BtB[0:(self.nBasis + self.prior.dfSpline), 0:(self.nBasis + self.prior.dfSpline)], self.Bty[0:(self.nBasis + self.prior.dfSpline)])
            
            fullRank = qf_cand != None
            if not fullRank:
                return
            
            sseCand = self.data.ssy - self.varCoefs / (1 + self.varCoefs) * qf_cand.qf
            
            logMH = (cand.logMHnActFeat + # Adjustment for Adaptive Nott-Kuk-Duc Proposal
                     logMHbdCand - self.logMHbd + # Probability of birth
                     -self.data.n/2 * (np.log(sseCand) - np.log(self.sse)) - self.prior.dfSpline/2 * np.log(self.varCoefs + 1) + # likelihood
                     np.log(self.prior.nRidgeMean/(self.nRidge + 1))) # prior

            if np.log(np.random.rand()) < logMH:
                self.cmod = True
                # note, BtB and Bty are already updated
                self.nRidge = self.nRidge + 1
                self.nBasis = self.nBasis + self.prior.dfSpline
                self.qf = qf_cand.qf
                self.sse = sseCand
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(self.R, np.identity(self.nBasis))
                self.count[0] = self.count[0] + 1
                self.logMHbd = logMHbdCand
                self.nAct[self.nRidge - 1] = cand.nAct
                self.feat[self.nRidge - 1, 0:(cand.nAct)] = cand.feat
                self.projDir[self.nRidge - 1, 0:(cand.nAct)] = cand.projDir
                self.knots[self.nRidge - 1, :] = cand.knots

                self.wnAct[cand.nAct - 1] = self.wnAct[cand.nAct - 1] + 1
                self.wnAct_norm = self.wnAct / sum(self.wnAct)
                self.wfeat[cand.feat] = self.wfeat[cand.feat] + 1
                self.wfeat_norm = self.wfeat / sum(self.wfeat)

                self.basis = np.append(self.basis, cand.basis, axis=1)


        elif move_type == 2:
            ## DEATH step

            tokill_ind = np.random.choice(self.nRidge)
            ind = list(range(self.nBasis))
            tokill_slice = slice(tokill_ind*self.prior.dfSpline + 1, (tokill_ind + 1)*self.prior.dfSpline + 1)
            del ind[tokill_slice]

            qf_cand = getQf(self.BtB[np.ix_(ind, ind)], self.Bty[ind])

            fullRank = qf_cand != None
            if not fullRank:
                return
            
            sseCand = self.data.ssy - self.varCoefs / (1 + self.varCoefs) * qf_cand.qf

            wnAct = self.wnAct.copy()
            wnAct[self.nAct[tokill_ind] - 1] = wnAct[self.nAct[tokill_ind] - 1] - 1
            wnAct_norm = wnAct / sum(wnAct)
            wfeat = self.wfeat.copy()
            wfeat[self.feat[tokill_ind, 0:self.nAct[tokill_ind]]] = wfeat[self.feat[tokill_ind, 0:self.nAct[tokill_ind]]] - 1
            wfeat_norm = wfeat / sum(wfeat)

            logMHnActFeat = getlogMHnActFeat(self.nAct[tokill_ind], self.feat[tokill_ind, 0:self.nAct[tokill_ind]], wnAct_norm, wfeat_norm, self.data.p, self.prior.nActMax)

            logMH = (logMHnActFeat + # Adjustment for Adaptive Nott-Kuk-Duc Proposal
                     logMHbdCand - self.logMHbd + # Probability of death
                     -self.data.n/2 * (np.log(sseCand) - np.log(self.sse)) + self.prior.dfSpline/2 * np.log(self.varCoefs + 1) + # likelihood
                     np.log(self.nRidge/self.prior.nRidgeMean)) # prior

            if np.log(np.random.rand()) < logMH:
                self.cmod = True
                self.nRidge = self.nRidge - 1
                self.nBasis = self.nBasis - self.prior.dfSpline
                self.qf = qf_cand.qf
                self.sse = sseCand
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(self.R, np.identity(self.nBasis))
                self.count[1] = self.count[1] + 1
                self.logMHbd = logMHbdCand

                self.Bty[0:self.nBasis] = self.Bty[ind]
                self.BtB[0:self.nBasis, 0:self.nBasis] = self.BtB[np.ix_(ind, ind)]

                temp = self.nAct[0:(self.nRidge + 1)]
                temp = np.delete(temp, tokill_ind)
                self.nAct = self.nAct * 0
                self.nAct[0:(self.nRidge)] = temp[:]

                temp = self.feat[0:(self.nRidge + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.feat = self.feat * 0
                self.feat[0:(self.nRidge), :] = temp[:]

                temp = self.projDir[0:(self.nRidge + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.projDir = self.projDir * 0
                self.projDir[0:(self.nRidge), :] = temp[:]

                temp = self.knots[0:(self.nRidge + 1), :]
                temp = np.delete(temp, tokill_ind, 0)
                self.knots = self.knots * 0
                self.knots[0:(self.nRidge), :] = temp[:]

                self.wnAct = wnAct[:]
                self.wnAct_norm = wnAct_norm[:]
                self.wfeat = wfeat[:]
                self.wfeat_norm = wfeat_norm[:]

                self.basis = np.delete(self.basis, tokill_slice, 1)

        else: # START HERE (try this step)
            ## CHANGE step

            tochange_ind = np.random.choice(self.nRidge)

            cand = genRidgeChange(self.projDir[tochange_ind, 0:self.nAct[tochange_ind]], self.prior.precProjDirProp,
                                  self.nAct[tochange_ind], self.feat[tochange_ind, 0:self.nAct[tochange_ind]],
                                  self.maxPropZero, self.prior.probRelu, self.quantKnots, self.data.X)

            CtC = np.dot(cand.basis.T, cand.basis)
            BtC = np.dot(self.basis.T, cand.basis)
            Cty = np.dot(cand.basis.T, self.data.y)

            ind = list(range(self.nBasis))
            tochange_slice = slice(tochange_ind*self.prior.dfSpline + 1, (tochange_ind + 1)*self.prior.dfSpline + 1)
            BtB_cand = self.BtB[np.ix_(ind, ind)].copy()
            BtB_cand[tochange_slice, :] = BtC.T
            BtB_cand[:,tochange_slice] = BtC
            BtB_cand[tochange_slice, tochange_slice] = CtC

            Bty_cand = self.Bty[0:self.nBasis].copy()
            Bty_cand[tochange_slice] = Cty

            qf_cand = getQf(BtB_cand, Bty_cand)

            fullRank = qf_cand != None
            if not fullRank:
                return
            
            sseCand = self.data.ssy - self.varCoefs / (1 + self.varCoefs) * qf_cand.qf
            
            logMH = -self.data.n/2 * (np.log(sseCand) - np.log(self.sse)) # likelihood

            if np.log(np.random.rand()) < logMH:
                self.cmod = True
                self.qf = qf_cand.qf
                self.sse = sseCand
                self.bhat = qf_cand.bhat
                self.R = qf_cand.R
                self.R_inv_t = sp.linalg.solve_triangular(self.R, np.identity(self.nBasis))  # check this
                self.count[2] = self.count[2] + 1

                self.Bty[0:self.nBasis] = Bty_cand
                self.BtB[0:self.nBasis, 0:self.nBasis] = BtB_cand

                self.projDir[tochange_ind, 0:self.nAct[tochange_ind]] = cand.projDir
                self.knots[tochange_ind, :] = cand.knots

                self.basis[:, tochange_slice] = cand.basis

        shapeVarResid = self.data.n / 2
        scaleVarResid = (self.data.ssy - np.dot(self.bhat.T, self.Bty[0:self.nBasis]) / (1 + self.varCoefs)) / 2
        if scaleVarResid < 0:
            scaleVarResid = 1.e-10
            
        self.sdResid = np.sqrt(1 / np.random.gamma(shapeVarResid, 1 / scaleVarResid, size=1))

        self.coefs = self.bhat * self.varCoefs / (1 + self.varCoefs) + np.dot(self.R_inv_t, np.random.normal(size=self.nBasis)) * np.sqrt(
            self.sdResid * self.varCoefs / (1 + self.varCoefs))

        temp = np.dot(self.R, self.coefs)
        qf2 = np.dot(temp, temp)
        shapeVarCoefs = self.prior.shapeVarCoefs + self.nBasis / 2
        scaleVarCoefs = self.prior.scaleVarCoefs + qf2 / self.nBasis / 2
        self.varCoefs = 1/np.random.gamma(shapeVarCoefs, 1 / scaleVarCoefs, size=1)



class bpprModel:
    """The model structure, including the current RJMCMC state and previous saved states; with methods for saving the
        state, plotting MCMC traces, and predicting"""
    def __init__(self, data, prior, nstore):
        """Get starting state, build storage structures"""
        self.data = data
        self.prior = prior
        self.state = bpprState(self.data, self.prior)
        self.nstore = nstore
        sdResid = np.zeros(nstore)
        varCoefs = np.zeros(nstore)
        nRidge = np.zeros(nstore, dtype=int)
        nRidge_models = np.zeros(nstore, dtype=int)
        nBasis_models = np.zeros(nstore, dtype=int)
        nAct = np.zeros([nstore, self.prior.nRidgeMax], dtype=int)
        feat = np.zeros([nstore, self.prior.nRidgeMax, self.prior.nActMax], dtype=int)
        projDir = np.zeros([nstore, self.prior.nRidgeMax, self.prior.nActMax])
        knots = np.zeros([nstore, self.prior.nRidgeMax, self.prior.dfSpline + 2])
        coefs = np.zeros([nstore, self.prior.dfSpline*self.prior.nRidgeMax + 1])
        self.samples = Samples(sdResid, varCoefs, nRidge, nRidge_models, nBasis_models, nAct, feat, projDir, knots, coefs)
        self.k = 0
        self.k_mod = -1
        self.model_lookup = np.zeros(nstore, dtype=int)
        return

    def writeState(self):
        """Take relevant parts of state and write to storage (only manipulates storage vectors created in init)"""
        self.samples.sdResid[self.k] = self.state.sdResid
        self.samples.varCoefs[self.k] = self.state.varCoefs
        self.samples.coefs[self.k, 0:self.state.nBasis] = self.state.coefs
        self.samples.nRidge[self.k] = self.state.nRidge

        if self.state.cmod: # basis part of state was changed
            self.k_mod = self.k_mod + 1
            self.samples.nRidge_models[self.k_mod] = self.state.nRidge
            self.samples.nBasis_models[self.k_mod] = self.state.nBasis
            self.samples.nAct[self.k_mod, 0:self.state.nRidge] = self.state.nAct[0:self.state.nRidge]
            self.samples.feat[self.k_mod, 0:self.state.nRidge, :] = self.state.feat[0:self.state.nRidge, :]
            self.samples.projDir[self.k_mod, 0:self.state.nRidge, :] = self.state.projDir[0:self.state.nRidge, :]
            self.samples.knots[self.k_mod, 0:self.state.nRidge, :] = self.state.knots[0:self.state.nRidge, :]
            self.state.cmod = False

        self.model_lookup[self.k] = self.k_mod
        self.k = self.k + 1

    def plot(self):
        """
        Trace plots and predictions/residuals

        * top left - trace plot of number of ridge functions (excluding burn-in and keepEveryning)
        * top right - trace plot of residual variance
        * bottom left - training data against predictions
        * bottom right - histogram of residuals (posterior mean) with assumed Gaussian overlaid.
        """
        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1)
        plt.plot(self.samples.nRidge)
        plt.ylabel("number of ridge functions")
        plt.xlabel("MCMC iteration (post-burn)")

        ax = fig.add_subplot(2, 2, 2)
        plt.plot(self.samples.sdResid)
        plt.ylabel("error sd")
        plt.xlabel("MCMC iteration (post-burn)")

        ax = fig.add_subplot(2, 2, 3)
        yhat = self.predict(self.data.X_orig).mean(axis=0)  # posterior predictive mean
        plt.scatter(self.data.y, yhat)
        abline(1, 0)
        plt.xlabel("observed")
        plt.ylabel("posterior prediction")

        ax = fig.add_subplot(2, 2, 4)
        plt.hist(self.data.y - yhat, color="skyblue", ec="white", density=True)
        axes = plt.gca()
        x = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
        plt.plot(x, sp.stats.norm.pdf(x, scale=np.sqrt(self.samples.sdResid.mean())), color='red')
        plt.xlabel("residuals")
        plt.ylabel("density")

        fig.tight_layout()

        plt.show()

    def makeAllBasis(self, model_ind, X):
        """Make basis matrix for model"""
        nRidge = self.samples.nRidge_models[model_ind]
        nBasis = self.samples.nBasis_models[model_ind]
        n = len(X)
        mat = np.zeros([n, nBasis])
        mat[:, 0] = 1
        start = 1
        stop = 1 + self.prior.dfSpline
        for j in range(nRidge):
            nAct = self.samples.nAct[model_ind, j]
            proj = np.dot(X[:, self.samples.feat[model_ind, j, 0:nAct]],
                          self.samples.projDir[model_ind, j, 0:nAct])
            mat[:, start:stop] = makeBasis(proj, self.samples.knots[model_ind, j, :])
            start = start + self.prior.dfSpline
            stop = stop + self.prior.dfSpline
        return mat

    def predict(self, X, mcmc_use=None, nugget=False):
        """
        BPPR prediction using new inputs (after training).

        :param X: matrix (numpy array) of predictors with dimension nxp, where n is the number of prediction points and
            p is the number of inputs (features). p must match the number of training inputs, and the order of the
            columns must also match.
        :param mcmc_use: which MCMC samples to use (list of integers of length m).  Defaults to all MCMC samples.
        :param nugget: whether to use the error variance when predicting.  If False, predictions are for mean function.
        :return: a matrix (numpy array) of predictions with dimension mxn, with rows corresponding to MCMC samples and
            columns corresponding to prediction points.
        """
        if len(X.shape)==1:
            X = X[None, :]

        Xs = normalize(X, self.data.Xmn, self.data.Xsd)
        if np.any(mcmc_use == None):
            mcmc_use = np.array(range(self.nstore))
        out = np.zeros([len(mcmc_use), len(Xs)])
        models = self.model_lookup[mcmc_use]
        umodels = set(models)
        k = 0
        for j in umodels:
            mcmc_use_j = mcmc_use[np.ix_(models == j)]
            nn = len(mcmc_use_j)
            out[range(k, nn + k), :] = np.dot(self.samples.coefs[mcmc_use_j, 0:self.samples.nBasis_models[j]],
                                              self.makeAllBasis(j, Xs).T)
            k = k + nn
        if nugget:
            out = out + np.random.normal(size=[len(Xs), len(mcmc_use)], scale=self.samples.sdResid[mcmc_use]).T
        return out


def bppr(X, y, nPost = 10000, nBurn = 9000, keepEvery = 1, nRidgeMean = 10, nActMax = 3, dfSpline = 4,
         probRelu = 2/3, wnAct = None, wfeat = None, shapeVarCoefs = None, scaleVarCoefs = None, minNonzero = None,
         scaleProjDirProp = None, nRidgeMax = None, verbose = True):
    """
    **Bayesian Projection Pursuit Regression - model fitting**

    This function takes training data, priors, and algorithmic constants and fits a BayesPPR model.  The result is a set of
    posterior samples of the model.  The resulting object has a predict function to generate posterior predictive
    samples.  Default settings of priors and algorithmic parameters should only be changed by users who understand
    the model.

    :param X: matrix (numpy array) of predictors of dimension nxp, where n is the number of training examples and p is
        the number of inputs (features).
    :param y: response vector (numpy array) of length n.
    :param nPost: total number of MCMC iterations (integer)
    :param nBurn: number of MCMC iterations to throw away as burn-in (integer, less than nPost).
    :param keepEvery: thin MCMC iterations and keep only every keepEvery (integer).
    :param nRidgeMean: prior mean number of ridge functions.
    :param nActMax: maximum numer of nonzero elements in each projection direction (integer, less than p)
    :param dfSpline: degrees of freedom for spline basis
    :param probRelu: prior probability that any given ridge function uses a relu transformation.
    :param wnAct: nominal weight for degree of interaction, used in generating candidate basis functions. Should be greater
        than 0.
    :param wfeat: nominal weight for features, used in generating candidate basis functions. Should be greater than 0.
    :param shapeVarCoefs: shape for inverse gamma prior on the variance of the coefficients.
    :param scaleVarCoefs: scale for inverse gamma prior on the variance of the coefficients.
    :param minNonzero: minimum number of non-zero points in a basis function. Defaults to 20 or 0.1 times the
        number of observations, whichever is smaller.
    :param scaleProjDirProp: scale parameter for generating proposed projection directions. Should be in (0, 1); default is about 0.002.
    :param nRidgeMax: maximum number of ridge functions (integer)
    :param verbose: boolean for printing progress
    :return: an object of class bpprModel, which includes predict and plot functions.
    """

    t0 = time.time()
    bd = bpprData(X, y)
    if bd.p < nActMax:
        nActMax = bd.p
        
    if wnAct == None:
        wnAct = np.ones(nActMax)
        
    if wfeat == None:
        wfeat = np.ones(bd.p)
        
    if shapeVarCoefs == None:
        shapeVarCoefs = 0.5
        
    if scaleVarCoefs == None:
        scaleVarCoefs = bd.n / 2
        
    if minNonzero == None:
        minNonzero = min(20, .1 * bd.n)
        
    if scaleProjDirProp == None:
        precProjDirProp = 1000  
    elif scaleProjDirProp > 1 or scaleProjDirProp <= 0:
        return
    
    else:
        temp = 1/scaleProjDirProp
        precProjDirProp = (temp - 1) + np.sqrt(temp * (temp - 1))

    if nRidgeMax == None:
        nRidgeMax = int(min(150, np.floor(bd.n/dfSpline) - 2))
        
    if nRidgeMax <= 0:
        return # dfSpline is too large compared to the sample size.

    bp = bpprPrior(nRidgeMean, nRidgeMax, nActMax, dfSpline, probRelu, shapeVarCoefs, scaleVarCoefs, minNonzero, precProjDirProp, wnAct, wfeat)
    nstore = int((nPost - nBurn) / keepEvery)
    bm = bpprModel(bd, bp, nstore)  # if we add tempering, bm should have as many states as temperatures
    for i in range(nPost): # rjmcmc loop
        bm.state.update()
        if i > (nBurn - 1) and ((i - nBurn + 1) % keepEvery) == 0:
            bm.writeState()
        if verbose and i % 500 == 0:
            print('\rBPPR MCMC {:.1%} Complete'.format(i / nPost), end='')
            # print(str(datetime.now()) + ', nRidge: ' + str(bm.state.nRidge))
    t1 = time.time()
    print('\rBPPR MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    # del bm.writeState # the user should not have access to this
    return bm


# class bpprBasis:
#     """Structure for functional response BPPR model using a basis decomposition, gets a list of BPPR models"""
#     def __init__(self, X, y, basis, newy, y_mean, y_sd, trunc_error, ncores=1, **kwargs):
#         """
#         Fit BPPR model with multivariate/functional response by projecting onto user specified basis.
# 
#         :param X: matrix (numpy array) of predictors of dimension nxp, where n is the number of training examples and
#             p is the number of inputs (features).
#         :param y: response matrix (numpy array) of dimension nxq, where q is the number of multivariate/functional
#             responses.
#         :param basis: matrix (numpy array) of basis functions of dimension qxk.
#         :param newy: matrix (numpy array) of y projected onto basis, dimension kxn.
#         :param y_mean: vector (numpy array) of length q with the mean if y was centered before obtaining newy.
#         :param y_sd: vector (numpy array) of length q with the standard deviation if y was scaled before obtaining newy.
#         :param trunc_error: numpy array of projection truncation errors (dimension qxn)
#         :param ncores: number of threads to use when fitting independent BPPR models (integer less than or equal to
#             npc).
#         :param kwargs: optional arguments to bppr function.
#         """
#         self.basis = basis
#         self.X = X
#         self.y = y
#         self.newy = newy
#         self.y_mean = y_mean
#         self.y_sd = y_sd
#         self.trunc_error = trunc_error
#         self.nRidge = len(basis[0])
# 
#         if ncores == 1:
#             self.bm_list = list(map(lambda ii: bppr(self.X, self.newy[ii, :], **kwargs), list(range(self.nRidge))))
#         else:
#             #with Pool(ncores) as pool: # this approach for pathos.multiprocessing
#             #    self.bm_list = list(
#             #        pool.map(lambda ii: bppr(self.X, self.newy[ii, :], **kwargs), list(range(self.nRidge))))
#             temp = PoolBPPR(self.X, self.newy, **kwargs)
#             self.bm_list = temp.fit(ncores, self.nRidge)
#         return
# 
#     def predict(self, X, mcmc_use=None, nugget=False, trunc_error=False, ncores=1):
#         """
#         Predict the functional response at new inputs.
# 
#         :param X: matrix (numpy array) of predictors with dimension nxp, where n is the number of prediction points and
#             p is the number of inputs (features). p must match the number of training inputs, and the order of the
#             columns must also match.
#         :param mcmc_use: which MCMC samples to use (list of integers of length m).  Defaults to all MCMC samples.
#         :param nugget: whether to use the error variance when predicting.  If False, predictions are for mean function.
#         :param trunc_error: whether to use truncation error when predicting.
#         :param ncores: number of cores to use while predicting (integer).  In almost all cases, use ncores=1.
#         :return: a numpy array of predictions with dimension mxnxq, with first dimension corresponding to MCMC samples,
#             second dimension corresponding to prediction points, and third dimension corresponding to
#             multivariate/functional response.
#         """
#         if ncores == 1:
#             pred_coefs = list(map(lambda ii: self.bm_list[ii].predict(X, mcmc_use, nugget), list(range(self.nRidge))))
#         else:
#             #with Pool(ncores) as pool:
#             #    pred_coefs = list(
#             #        pool.map(lambda ii: self.bm_list[ii].predict(X, mcmc_use, nugget), list(range(self.nRidge))))
#             temp = PoolBPPRPredict(X, mcmc_use, nugget, self.bm_list)
#             pred_coefs = temp.predict(ncores, self.nRidge)
#         out = np.dot(np.dstack(pred_coefs), self.basis.T)
#         out2 = out * self.y_sd + self.y_mean
#         if trunc_error:
#             out2 += self.trunc_error[:, np.random.choice(np.arange(self.trunc_error.shape[1]), size=np.prod(out.shape[:2]), replace=True)].reshape(out.shape)
#         return out2
# 
#     def plot(self):
#         """
#         Trace plots and predictions/residuals
# 
#         * top left - trace plot of number of basis functions (excluding burn-in and keepEveryning) for each BPPR model
#         * top right - trace plot of residual variance for each BPPR model
#         * bottom left - training data against predictions
#         * bottom right - histogram of residuals (posterior mean).
#         """
# 
#         fig = plt.figure()
# 
#         ax = fig.add_subplot(2, 2, 1)
#         for i in range(self.nRidge):
#             plt.plot(self.bm_list[i].samples.nRidge)
#         plt.ylabel("number of basis functions")
#         plt.xlabel("MCMC iteration (post-burn)")
# 
#         ax = fig.add_subplot(2, 2, 2)
#         for i in range(self.nRidge):
#             plt.plot(self.bm_list[i].samples.sdResid)
#         plt.ylabel("error variance")
#         plt.xlabel("MCMC iteration (post-burn)")
# 
#         ax = fig.add_subplot(2, 2, 3)
#         yhat = self.predict(self.bm_list[0].data.X_orig).mean(axis=0)  # posterior predictive mean
#         plt.scatter(self.y, yhat)
#         abline(1, 0)
#         plt.xlabel("observed")
#         plt.ylabel("posterior prediction")
# 
#         ax = fig.add_subplot(2, 2, 4)
#         plt.hist((self.y - yhat).reshape(np.prod(yhat.shape)), color="skyblue", ec="white", density=True)
#         plt.xlabel("residuals")
#         plt.ylabel("density")
# 
#         fig.tight_layout()
# 
#         plt.show()
# 
# class bpprPCAsetup:
#     """
#     Wrapper to get principal components that would be used for bpprPCA.  Mainly used for checking how many PCs should be used.
# 
#     :param y: response matrix (numpy array) of dimension nxq, where n is the number of training examples and q is the number of multivariate/functional
#         responses.
#     :param npc: number of principal components to use (integer, optional if percVar is specified).
#     :param percVar: percent (between 0 and 100) of variation to explain when choosing number of principal components
#         (if npc=None).
#     :param center: whether to center the responses before principal component decomposition (boolean).
#     :param scale: whether to scale the responses before principal component decomposition (boolean).
#     :return: object with plot method.
#     """
#     def __init__(self, y, center=True, scale=False):
#         self.y = y
#         self.y_mean = 0
#         self.y_sd = 1
#         if center:
#             self.y_mean = np.mean(y, axis=0)
#         if scale:
#             self.y_sd = np.std(y, axis=0)
#             self.y_sd[self.y_sd == 0] = 1
#         self.y_scale = np.apply_along_axis(lambda row: (row - self.y_mean) / self.y_sd, 1, y)
#         #decomp = np.linalg.svd(y_scale.T)
#         U, s, V = np.linalg.svd(self.y_scale.T)
#         self.evals = s ** 2
#         self.basis = np.dot(U, np.diag(s))
#         self.newy = V
#         return
#     
#     def plot(self, npc=None, percVar=None):
#         """
#         Plot of principal components, eigenvalues
# 
#         * left - principal components; grey are excluded by setting of npc or percVar
#         * right - eigenvalues (squared singular values), colored according to principal components
#         """
# 
#         cs = np.cumsum(self.evals) / np.sum(self.evals) * 100.
# 
#         if npc == None and percVar == 100:
#             npc = len(self.evals)
#         if npc == None and percVar is not None:
#             npc = np.where(cs >= percVar)[0][0] + 1
#         if npc == None or npc > len(self.evals):
#             npc = len(self.evals)
# 
#         fig = plt.figure()
# 
#         cmap = plt.get_cmap("tab10")
# 
#         ax = fig.add_subplot(1, 2, 1)
#         if npc < len(self.evals):
#             plt.plot(self.basis[:, npc:], color='grey')
#         for i in range(npc):
#             plt.plot(self.basis[:, i], color=cmap(i%10))
#         plt.ylabel("principal components")
#         plt.xlabel("multivariate/functional index")
# 
#         ax = fig.add_subplot(1, 2, 2)
#         x = np.arange(len(self.evals)) + 1
#         if npc < len(self.evals):
#             plt.scatter(x[npc:], cs[npc:], facecolors='none', color='grey')
#         for i in range(npc):
#             plt.scatter(x[i], cs[i], facecolors='none', color=cmap(i%10))
#         plt.axvline(npc)
#         #if percVar is not None:
#         #    plt.axhline(percVar)
#         plt.ylabel("cumulative eigenvalues (percent variance)")
#         plt.xlabel("index")
# 
#         fig.tight_layout()
# 
#         plt.show()
# 
# def bpprPCA(X, y, npc=None, percVar=99.9, ncores=1, center=True, scale=False, **kwargs):
#     """
#     Wrapper to get principal components and call bpprBasis, which then calls bppr function to fit the BPPR model for
#     functional (or multivariate) response data.
# 
#     :param X: matrix (numpy array) of predictors of dimension nxp, where n is the number of training examples and p is
#         the number of inputs (features).
#     :param y: response matrix (numpy array) of dimension nxq, where q is the number of multivariate/functional
#         responses.
#     :param npc: number of principal components to use (integer, optional if percVar is specified).
#     :param percVar: percent (between 0 and 100) of variation to explain when choosing number of principal components
#         (if npc=None).
#     :param ncores: number of threads to use when fitting independent BPPR models (integer less than or equal to npc).
#     :param center: whether to center the responses before principal component decomposition (boolean).
#     :param scale: whether to scale the responses before principal component decomposition (boolean).
#     :param kwargs: optional arguments to bppr function.
#     :return: object of class bpprBasis, with predict and plot functions.
#     """
# 
#     setup = bpprPCAsetup(y, center, scale)
# 
#     if npc == None:
#         cs = np.cumsum(setup.evals) / np.sum(setup.evals) * 100.
#         npc = np.where(cs > percVar)[0][0] + 1
# 
#     if ncores > npc:
#         ncores = npc
# 
#     basis = setup.basis[:, :npc]
#     newy = setup.newy[:npc, :]
#     trunc_error = np.dot(basis, newy) - setup.y_scale.T
# 
#     print('\rStarting bpprPCA with {:d} components, using {:d} cores.'.format(npc, ncores))
# 
#     return bpprBasis(X, y, basis, newy, setup.y_mean, setup.y_sd, trunc_error, ncores, **kwargs)
# 

######################################################
## test it out

if __name__ == '__main__':

    if True:
        def f(x):
            out = 10.0 * np.sin(2*np.pi * x[:, 0] * x[:, 1]) + 20.0 * (x[:, 2] - 0.5) ** 2 + 10.0 * x[:, 3] + 5.0 * x[:, 4]
            return out


        n = 500
        p = 10
        x = np.random.rand(n, p)
        X = np.random.rand(1000, p)
        y = f(x) + np.random.normal(size=n)

        mod = bppr(x, y, nPost=10000, nBurn=9000)
        pred = mod.predict(X, mcmc_use = np.array([1, 100]), nugget = False)

        mod.plot()

        print(np.var(mod.predict(X).mean(axis=0)-f(X)))

#     if False:
#         def f2(x):
#             out = 10. * np.sin(np.pi * tt * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]
#             return out
# 
# 
#         tt = np.linspace(0, 1, 50)
#         n = 500
#         p = 9
#         x = np.random.rand(n, p) - .5
#         X = np.random.rand(1000, p) - .5
#         e = np.random.normal(size=n * len(tt))
#         y = np.apply_along_axis(f2, 1, x)  # + e.reshape(n,len(tt))
# 
#         modf = bpprPCA(x, y, ncores=2, percVar=99.99)
#         modf.plot()
# 
#         pred = modf.predict(X, mcmc_use=np.array([1,100]), nugget=True)
# 
#         ind = 11
#         plt.plot(pred[:,ind,:].T)
#         plt.plot(f2(X[ind,]),'bo')
# 
#         plt.plot(np.apply_along_axis(f2, 1, X), np.mean(pred,axis=0))
#         abline(1,0)
