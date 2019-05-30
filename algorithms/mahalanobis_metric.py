# Perform mirror descent for metric learning given labeled data triplets. MDML learns a Mahalanobis matrix M = L*L',
# which is possibly low rank, and positive semi-definite. The space spanned by L is low-dimensional and captures the
# separation of the data best. The approach can handle multi-class data sets by appropriately setting labels y = 1
# (similar) or y = -1 (dissimilar).

import numpy as np
import warnings
from timeit import default_timer as timer
from optimization.eigen_values import rational_rank_one_update


def learn(x, example_pairs, similarity_labels, eta=1.0, rho=1.0, loss='hinge', bregman_function='Frobenius', tol=1e-12,
          algorithm='rational', verbose=False, debug=False):
    # TODO: Check dimensions and perform sanity checks

    # Initialize
    (n, d) = x.shape
    max_iterations = example_pairs.shape[0]
    mu = 1.0
    v = np.eye(d)
    e = np.ones((d,))
    start_time = timer()
    num_updates = 0

    if verbose:
        print('Performing metric learning with {0} + {1}'.format(bregman_function, loss))
        print('-------------+--------------+--------------+---------------+--------')
        print('     Iter    |    margin    |    alpha     |       mu      | zeros')
        print('-------------+--------------+--------------+---------------+--------')

    # Start the updates
    for t in range(0, max_iterations):
        # Get the indices of the current pair of points
        i = example_pairs[t, 0]
        j = example_pairs[t, 1]

        # Compute hinge loss gradients using the full eigen-decomposition, i.e., loss(Mt)
        u = x[i, :] - x[j, :]                                 # Compute the difference between the examples
        w = v.T @ u                                           # Compute rank-one vector and project into eigen-space
        margin = similarity_labels[t] * (mu - w.T @ (np.diag(e) @ w))  # Compute margin for this pair of examples

        # Set function to compute (most of) the gradient of the loss function. This function computes a scalar a(m(t)),
        # which is a function of the margin at num_updates t, m(t). The gradient of the loss can be expressed as below,
        #       dL / d M  =  a(m(t))*y(t)*(x(t) - z(t))*(x(t) - z(t))', and
        #       dL / d mu = -a(m(t))*y(t)
        # given a pair of points at num_updates t: x(t) and z(t), with label y(t); here alpha = a(m(t))*y(t)
        if loss is 'hinge':
            alpha = np.heaviside(1 - margin, 0.0) * similarity_labels[t]
        elif loss is 'squared':
            alpha = np.max([1 - margin, 0.0]) * similarity_labels[t]
        elif loss is 'exponential':
            alpha = np.exp(-margin) * similarity_labels[t]
        elif loss is 'logistic':
            alpha = np.exp(-margin) / (1 + np.exp(-margin)) * similarity_labels[t]
        else:
            alpha = 0.0

        # Do not update if the loss is zero
        if np.abs(alpha) < tol:
            continue

        # Update the Mahalanobis matrix: [Step 1] Restrict the update to the range space of the Mahalanobis matrix
        nz = (np.abs(e) > tol)
        if bregman_function is 'vonNeumann':
            e[nz] = np.log(e[nz])
        elif bregman_function is 'Burg':
            e[nz] = -1 / e[nz]

        # Update the Mahalanobis matrix: [Step 2]  Compute the rank-one update in the Bregman space
        eta_t = eta / np.sqrt(t + 1)  # The learning rate for this iteration is eta / sqrt(t)

        if algorithm is 'explicit':
            m = v @ np.diag(e) @ v.T + eta_t * alpha * np.outer(u, u)  # Explicitly compute the update
            m = (m + m.T) / 2.0                                 # Make updated matrix symmetric for numerical stability
            e, v = np.linalg.eigh(m)                            # Compute the eigen-decomposition using Numpy

        elif algorithm is 'rational':
            # Perform the update matrix explicitly
            if debug:
                m = v @ np.diag(e) @ v.T + eta_t * alpha * np.outer(u, u)  # Explicitly compute the update
                m = (m + m.T) / 2.0  # Make updated matrix symmetric for numerical stability

            # Perform the update using rational interpolation
            e, v, diagnostic = rational_rank_one_update(v, e, w, eta_t * alpha, tol)

            # Compute the reconstruction error to ensure that the rational interpolation succeeded
            if debug:
                err = np.linalg.norm(m - v @ np.diag(e) @ v.T, ord=2)
                if err > np.sqrt(tol):
                    warnings.warn('High reconstruction error = {0}'.format(err), RuntimeWarning)

        # Update the Mahalanobis matrix: [Step 3]  Shrink/threshold the eigen-values appropriately
        e[e < rho] = 0.0
        num_updates = num_updates + 1

        # Update the Mahalanobis matrix: [Step 4]  Return to original space
        nz = (np.abs(e) > tol)
        if bregman_function is 'vonNeumann':
            e[nz] = np.exp(e[nz])
        elif bregman_function is 'Burg':
            e[nz] = -1 / e[nz]

        # Update the bias term: [single step]
        if bregman_function is 'Frobenius':
            mu = np.max([mu + eta_t * alpha, 1.0])
        elif bregman_function is 'vonNeumann':
            mu = np.max([np.exp(np.log(mu) + eta_t * alpha), 1.0])
        elif bregman_function is 'Burg':
            mu = np.max([1.0 / (1.0 / mu + eta_t * alpha), 1.0])

        if verbose:
            num_zeros = d - np.count_nonzero(e)
            print('{0:7d} ({1:+1d}) | {2:+.4e}  |  {3:+.4e} |   {4:.4e}  | {5:4d}'.format(t, similarity_labels[t],
                                                                                        margin, alpha, mu, num_zeros))

        if debug and t % 100 == 0 and t > 0:
            print('-------------+--------------+--------------+---------------+--------')

        # In debugging mode, we check for the most common algorithmic failures of
        if debug:
            if np.any(np.isnan(e)):
                raise RuntimeError('Have Nans in E at iteration {0}'.format(t))

            if np.any(np.isnan(v)) or np.any(v > 1 / tol):
                raise RuntimeError('Have NaNs or infs in V at num_updates {0}. Potential problems in the '
                                   'eigenvalue solver!'.format(t))

            if np.linalg.norm(v.T @ v - np.eye(d), ord=2) > np.sqrt(tol):
                raise RuntimeError('Lost orthogonality of V in num_updates in iteration {0}.'.format(t))

            if np.all(e < tol):
                raise RuntimeWarning('All eigenvalues became zero after num_updates {0}'.format(t - 1))

    # Finish debugging
    if verbose:
        print('-----------+--------------+--------------+---------------+--------')

    print('Num pairs of instances seen = {0}, Num updates = {1}.'.format(max_iterations, num_updates))
    end_time = timer()
    total_time = end_time - start_time
    return e, v, mu, total_time, num_updates
