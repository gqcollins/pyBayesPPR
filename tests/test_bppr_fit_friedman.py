import pyBayesPPR as pb
import numpy as np
from scipy import stats

# def test_bppr_fit():
    # Friedman function (Friedman, 1991, Multivariate Adaptive Regression Splines)
    # def f(x):
    #     return (10.0 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20.0 * (x[:, 2] - 0.5) ** 2
    #             + 10.0 * x[:, 3] + 5.0 * x[:, 4])

    # Set random seed for reproducibility.
    # np.random.seed(0)

    # # Generate data.
    # n = 500  # sample size
    # p = 10  # number of predictors (only 5 are used)
    # x = np.random.rand(n, p)  # predictors (training set)
    # y = f(x) + np.random.randn(n) * 0.1  # response (training set) with noise.
# B = 100
# rmse_train_python = []
# rmse_test_python = []
# for i in range(B):
#     # fit BPPR model with RJMCMC
#     X = np.genfromtxt('/Users/gqcolli/X_friedman.csv', delimiter=',', skip_header=True)
#     fX = np.genfromtxt('/Users/gqcolli/fX_friedman.csv', delimiter=',', skip_header=True)
#     mod = pb.bppr(X, fX)
#     preds = mod.predict(X)
#     rmse_train_python.append(np.sqrt(np.mean((preds.mean(0) - fX) ** 2)))

#     Xtest = np.genfromtxt('/Users/gqcolli/Xtest_friedman.csv', delimiter=',', skip_header=True)
#     fXtest = np.genfromtxt('/Users/gqcolli/fXtest_friedman.csv', delimiter=',', skip_header=True)
#     preds = mod.predict(Xtest)
#     rmse_test_python.append(np.sqrt(np.mean((preds.mean(0) - fXtest) ** 2)))


# np.savetxt("/Users/gqcolli/rmse_train_friedman_python.csv", rmse_train_python, delimiter=",")
# np.savetxt("/Users/gqcolli/rmse_test_friedman_python.csv", rmse_test_python, delimiter=",")

rmse_train_python = np.genfromtxt('/Users/gqcolli/rmse_train_friedman_python.csv', delimiter=',', skip_header=True)
rmse_test_python = np.genfromtxt('/Users/gqcolli/rmse_test_friedman_python.csv', delimiter=',', skip_header=True)

rmse_train_R = np.genfromtxt('/Users/gqcolli/rmse_train_friedman.csv', delimiter=',', skip_header=True)
rmse_test_R = np.genfromtxt('/Users/gqcolli/rmse_test_friedman.csv', delimiter=',', skip_header=True)

Spooled_train = np.sqrt(np.var(np.log(rmse_train_python))/len(rmse_train_python) + np.var(np.log(rmse_train_R))/len(rmse_train_R))
t_train = (np.mean(np.log(rmse_train_python)) - np.mean(np.log(rmse_train_R))) / Spooled_train
2*(1 - stats.t(df=len(rmse_train_python) + len(rmse_train_R) - 2).cdf(t_train))

Spooled_test = np.sqrt(np.var(np.log(rmse_test_python))/len(rmse_test_python) + np.var(np.log(rmse_test_R))/len(rmse_test_R))
t_test = (np.mean(np.log(rmse_test_python)) - np.mean(np.log(rmse_test_R))) / Spooled_test
2*(stats.t(df=len(rmse_test_python) + len(rmse_test_R) - 2).cdf(t_test))
