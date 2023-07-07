import pyBayesPPR as pb
import numpy as np

def test_bppr_fit():
    # Gramacy-Lee (Gramacy & Lee, 2009)
    def f(x):
        return (np.exp(np.sin((0.9 * (x[:, 1] + 0.48))**10)) + x[:, 2] * x[:, 3] + x[:, 4])

    # Set random seed for reproducibility.
    np.random.seed(0)

    # Generate data.
    n = 500  # sample size
    p = 6  # number of predictors (only 5 are used)
    x = np.random.rand(n, p)  # predictors (training set)
    y = f(x) + np.random.randn(n) * 0.05  # response (training set) with noise.

    # fit BPPR model with RJMCMC
    mod = pb.bppr(x, y)

    # predict at new inputs (xnew)
    xnew = np.random.rand(1000, p)
    pred = mod.predict(xnew, nugget=True)

    # True values at new inputs.
    ynew = f(xnew)

    # Root mean squred error
    rmse = np.sqrt(np.mean((pred.mean(0) - ynew) ** 2))
    
    # Test that RMSE is less than 0.3 for this model, which should be the case
    # from previous tests.
    assert rmse < 0.4

