import numpy as np
import pandas as pd

def get_coefficients(design_matrix, response_vector, epsilon=.001):
    """
    Determine Logistic Regression coefficents using Fisher Scoring algorithm.
    Iteration ceases once changes between elements in coefficent matrix across
    consecutive iterations is less than epsilon.
    # =========================================================================
    # design_matrix      `X`     => n-by-(p+1)                                |
    # response_vector    `y`     => n-by-1                                    |
    # probability_vector `p`     => n-by-1                                    |
    # weights_matrix     `W`     => n-by-n                                    |
    # epsilon                    => threshold above which iteration continues |
    # =========================================================================
    # n                          => # of observations                         |
    # (p + 1)                    => # of parameterss, +1 for intercept term   |
    # =========================================================================
    # U => First derivative of Log-Likelihood with respect to                 |
    #      each beta_i, i.e. `Score Function`: X_transpose * (y - p)          |
    #                                                                         |
    # I => Second derivative of Log-Likelihood with respect to                |
    #      each beta_i. The `Information Matrix`: (X_transpose * W * X)       |
    #                                                                         |
    # X^T*W*X results in a (p+1)-by-(p+1) matrix                              |
    # X^T(y - p) results in a (p+1)-by-1 matrix                               |
    # (X^T*W*X)^-1 * X^T(y - p) results in a (p+1)-by-1 matrix                |
    # ========================================================================|
    """
    X = np.matrix(design_matrix)
    y = np.matrix(response_vector)

    # initialize logistic function used for Scoring calculations =>
    def pi_i(v): return (np.exp(v) / (1 + np.exp(v)))

    # initialize beta_0, p_0, W_0, I_0 & U_0 =>
    beta_0 = np.matrix(np.zeros(np.shape(X)[1])).T
    p_0 = pi_i(X * beta_0)
    W_pre = (np.array(p_0) * np.array(1 - p_0))
    W_0 = np.matrix(np.diag(W_pre[:, 0]))
    I_0 = X.T * W_0 * X
    U_0 = X.T * (y - p_0)

    # initialize variables for iteration =>
    beta_old = beta_0
    iter_I = I_0
    iter_U = U_0
    iter_p = p_0
    iter_W = W_0
    fisher_scoring_iterations = 0


    # iterate until abs(beta_new - beta_old) < epsilon =>
    while True:

        # Fisher Scoring Update Step =>
        coeffs.append(np.array(beta_old))
        fisher_scoring_iterations += 1
        beta_new = beta_old + iter_I.I * iter_U

        if all(np.abs(np.array(beta_new)-np.array(beta_old)) < epsilon):
            model_parameters  = beta_new
            fitted_values     = pi_i(X * model_parameters)
            covariance_matrix = iter_I.I
            break

        else:
            iter_p     = pi_i(X * beta_new)
            iter_W_pre = (np.array(iter_p) * np.array(1 - iter_p))
            iter_W     = np.matrix(np.diag(iter_W_pre[:, 0]))
            iter_I     = X.T * iter_W * X
            iter_U     = X.T * (y - iter_p)
            beta_old   = beta_new

    summary = {
        'model_parameters' : np.array(model_parameters),
        'fitted_values'    : np.array(fitted_values),
        'covariance_matrix': np.array(covariance_matrix),
        'number_iterations': fisher_scoring_iterations
    }

    return (summary)