import numpy as np
from pyBayesPPR import bppr 

def f(X): # Friedman function
    return (
        10. * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20. * 
        (X[:, 2] - .5) ** 2 + 10 * X[:, 3] + 5. * X[:, 4]
    )

# Set random seed for reproducibility.
np.random.seed(0)

# Generate data.
n = 500  # sample size
p = 10  # number of predictors (only 5 are used)
X = np.random.rand(n, p)  # predictors (training set)
y = np.random.normal(f(X), 1)  # response (training set) with noise.

# fit BayesPPR model with RJMCMC
mod = bppr(X, y)
mod.plot()
mod.traceplot()
mod.sobol()
mod.plot_sobol()

# predict at new inputs (Xnew)
X_test = np.random.rand(10000, p)
y_test = np.random.normal(f(X_test), 1)

pred = mod.predict(X_test)
mod.plot(X_test, y_test)
