import numpy as np
from timeit import default_timer as timer
import warnings


def sanity_check(v, e, t, rho):
    # If E is a diagonal matrix, make it a vector
    if len(e.shape) == 2:
        e = np.diagonal(e)
    elif len(e.shape) > 2:
        raise ValueError('Eigen-value structure (e) is not two-dimensional')

    # In order to circumvent Python being too-clever-by-half, we explicitly force all inputs to be floats
    v = np.float64(v)
    e = np.float64(e)
    t = np.float64(t)
    rho = np.float64(rho)

    return v, e, t, rho


# Given a matrix A = V diag(e) V' with known eigenvalues e, this function efficiently calculates the eigenvalues of the
# perturbed matrix, Ap = (A + rho*u*u'), where rho is some scalar and u is a vector.
def naive_rank_one_update(v, e, u, rho):
    v, e, u, rho = sanity_check(v, e, u, rho)            # Make sure that inputs are good
    a_new = v @ np.diag(e) @ v.T + rho * np.outer(u, u)  # Perform rank-one update directly
    a_new = (a_new + a_new.T) / 2                        # Ensure numerically stable symmetry
    v, e = np.linalg.eigh(a_new)                         # Decompose the updated matrix using Numpy
    return v, e


# Given a matrix A = V diag(e) V' with known eigenvalues e, this function efficiently calculates the eigenvalues of the
# perturbed matrix, Ap = (A + rho*u*u'), where rho is some scalar and u is a vector.
#
# NOTE: The algorithm takes t = V'*u as the input rather than z directly, i.e., we assume that Ap is expressed as
#               Ap = V*(diag(E) + rho*t*t')*V'
def rational_rank_one_update(v, e, t, rho, tol=1e-12):
    # Make sure that inputs are good
    v, e, u, rho = sanity_check(v, e, t, rho)

    # Initialize
    n = len(e)                  # Problem dimension
    gt = 10                     # Constant to test if fractions of t can be set to zero
    start_time = timer()
    e_index = np.argsort(e)     # Get the index of sorted eigenvalues
    e = np.sort(e)              # Sort the eigenvalues in ascending order
    v = v[:, e_index]           # Sort the eigen-vectors
    t = t[e_index]              # Sort the perturbation vector
    t_norm = np.linalg.norm(t)  # Norm of t
    e_max = np.max(np.fabs(e))  # Biggest eigen-value
    e[np.fabs(e) < n * tol * np.sqrt(e_max)] = 0.0  # Remove very small eigen-values

    # Now, get the repeated eigenvalues, and replace the corresponding eigen-vectors V(E) with V(E)*H, where H is a
    # Householder matrix designed to introduce sparsity into the linear combination space i.e., make t look like
    # [a, 0, 0, b, 0, c, 0, 0, 0, ...., 0]'
    for eig in np.unique(e):
        # Determine where the array starts and s
        e_index = np.nonzero(e == eig)[0]
        e_mult = len(e_index)

        # If this is a multiple eigenvalue, zero out all the components of t
        # but one corresponding to the one closest to the next highest (if rho
        # > 0, lowest if rho < 0)
        if e_mult > 1:
            w = t[e_index]
            w_norm = np.linalg.norm(w, ord=2)

            if rho < 0:
                # Set  t(e_index) = [-norm(w); zeros(e_mult-1, 1)]
                t[e_index[0]] = -w_norm
                t[e_index[1:]] = np.zeros((e_mult - 1,))
                w[0] += w_norm
            else:
                # Set  t(e_index) = [zeros(e_mult-1, 1); -norm(w)]
                t[e_index[:-1]] = np.zeros((e_mult - 1,))
                t[e_index[-1]] = -w_norm
                w[-1] += w_norm

            # Sometimes, some values slip through despite the accuracy check above. Make sure that the eigen-vector
            # calculation remains numerically backward stable
            w_norm = np.linalg.norm(w, ord=2)
            if w_norm > gt * (e_max + t_norm * t_norm) * tol:
                v[:, e_index] -= 2 * v[:, e_index] @ np.outer(w, w.T) / (w_norm * w_norm)

    # Remove very small values of t
    t_cutoff = gt * ((e_max / t_norm) + t_norm) * tol
    t[np.fabs(t) <= t_cutoff] = 0

    t_nonzero = np.fabs(t[:]) > t_cutoff
    e_bar = e[t_nonzero]
    t_bar = t[t_nonzero]
    t_size = len(t_bar)

    # Compute the all the eigenvalues, and the eigen-vectors corresponding to the components t(i) <> 0. The components
    # for t(i) == 0 remain unchanged
    mu = np.zeros((t_size, ))
    avg_iterations = 0
    avg_eigval_time = 0

    if rho > 0:
        for i in range(0, t_size):
            mu[i], iters, time = single_update(i, e_bar, t_bar, rho, tol)
            avg_iterations += iters
            avg_eigval_time += time
    else:
        for i in range(0, t_size):
            mu[i], iters, time = single_update(t_size-i-1, -e_bar[::-1], t_bar[::-1], -rho, tol)
            avg_iterations += iters
            avg_eigval_time += time

    e_new = e_bar + rho*mu
    e_new[np.fabs(e_new) < tol] = 0      # Clean up e_new
    e[t_nonzero] = e_new                 # Insert the new eigenvalues of the perturbed matrix

    # Compute the eigen-vectors using t
    v_new = np.zeros((n, t_size))
    for i in range(0, t_size):
        # If there is no numerical difference between one of the updated eigen-values and the old eigen-values
        if np.any(np.abs(e_new[i] - e_bar) < tol):
            # TODO: This shouldn't happen but still does sometimes; needs to be explored further.
            v_new[:, i] = v[:, t_nonzero][:, i]
            continue
        v_new[:, i] = v[:, t_nonzero] @ np.diag(1 / (e_new[i] - e_bar)) @ t[t_nonzero]
        v_new[:, i] /= np.linalg.norm(v_new[:, i], ord=2)
    v[:, t_nonzero] = v_new

    # Collect status and execution results into a diagnostic
    end_time = timer()
    diagnostic = {'total_time': end_time - start_time,
                  'avg_time': avg_eigval_time / t_size,
                  'avg_iterations': avg_iterations / t_size,
                  'tolerance': tol}

    return e, v, diagnostic


def single_update(i, eigs, t, rho, tol=1e-12, max_iterations=5000):
    start_time = timer()

    # Compute delta and initialize some looping variables
    # delta = [(e - eigs[i]) / rho for e in eigs]  # (E - E(i)) / rho
    delta = (eigs - eigs[i]) / rho
    n = len(delta)
    converged = False
    iteration = 0

    # The case for i = n is different, since many terms drop out of the interpolation expression making its computation
    # easier write this particular iteration separately
    if i == n-1:
        mu = 1  # Initialize the value of mu
    
        while not converged and iteration < max_iterations:
            iteration = iteration + 1

            # Compute some helper values
            psi, phi, d_psi, _ = evaluate_rationals(mu, delta, t, i, tol)
            mu += (1 + psi) / d_psi * psi
            w = 1 + phi + psi

            # Sanity-check the mu values
            mu = guard_values(mu, np.inf, tol, iteration)

            # Check for convergence
            if abs(w) <= tol * n * (1 + abs(psi) + abs(phi)):
                converged = True

    else:
        # Initialize the value of mu
        t_index = np.concatenate([np.arange(0, i), np.arange(i+2, n)])
        t_rest = t[t_index] @ (t[t_index] / (delta[t_index] - delta[i+1]))

        b = delta[i+1] + (t[i]**2 + t[i+1]**2) / (1 + t_rest)
        c = (delta[i+1] * t[i]**2) / (1 + t_rest)
        mu1 = b/2 - np.sqrt(b**2 - 4*c) / 2
        mu2 = b/2 + np.sqrt(b**2 - 4*c) / 2

        # Pick the smallest value for mu > 0
        if mu1 < tol < mu2:
            mu = mu2
        elif mu2 < tol < mu1:
            mu = mu1
        else:
            mu = min(abs(mu1), abs(mu2))

        # Now iterate by constructing interpolating rationals
        while not converged and iteration < max_iterations:
            iteration = iteration + 1

            # Compute some helper values
            delta_diff = delta[i+1] - mu
            psi, phi, d_psi, d_phi = evaluate_rationals(mu, delta, t, i, tol)

            c = 1 + phi - delta_diff * d_phi
            a = (delta_diff * (1 + phi) + psi**2/d_psi) / c + psi/d_psi
            w = 1 + phi + psi
            b = (delta_diff * w * psi) / (d_psi * c)

            # Update mu
            mu += 2*b / (a + np.sqrt(a**2 - 4*b))

            # Sanity-check the mu values
            mu = guard_values(mu, delta[i+1], tol, iteration)

            # Check for convergence
            if abs(w) <= tol * n * (1 + abs(psi) + abs(phi)):
                converged = True

    end_time = timer()
    exec_time = end_time - start_time

    if iteration >= max_iterations:
        warning_message = '\nMaximum number of iterations exceeded.'.format(iteration)
        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

    return mu, iteration, exec_time


def evaluate_rationals(x, delta, u, i, tol):
    psi = u[:i+1] @ (u[:i+1] / (delta[:i+1] - x))
    phi = u[i+1:] @ (u[i+1:] / (delta[i+1:] - x))

    d_psi = u[:i+1] @ (u[:i+1] / ((delta[:i+1] - x) ** 2))
    d_phi = u[i+1:] @ (u[i+1:] / ((delta[i+1:] - x) ** 2))

    if np.isinf(psi):
        psi = 1/tol
    if np.isinf(d_psi):
        d_psi = 1/tol
    if np.isinf(phi):
        phi = 1/tol
    if np.isinf(d_phi):
        d_phi = 1/tol

    return psi, phi, d_psi, d_phi


def guard_values(mu, delta, tol, iteration):

    if np.isinf(mu) or np.isnan(mu):
        mu = 1 / tol
        warning_message = '\npsi/phi computation failed. fixed mu = 1/tol. (Iter = {0}).'.format(iteration)
        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

    elif mu < 0:
        mu = tol
        warning_message = '\nComputed mu < 0: CORRECTED mu = +tol. (Iter = {0}).'.format(iteration)
        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

    elif mu > delta:
        mu = delta - tol
        warning_message = '\nComputed mu > delta(i+1): CORRECTED mu = delta(i+1)-tol.'.format(iteration)
        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

    return mu
