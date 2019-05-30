import numpy as np
from scipy.linalg import schur
from timeit import default_timer as timer
from optimization.eigen_values import rational_rank_one_update as eig_update
import warnings


def sanity_check(u, s, v, a, b, rho):
    # Ensure that S is a matrix
    if not len(s.shape) == 2:
        raise ValueError('Singular-value structure (s) is not two-dimensional')

    # In order to circumvent Python being too-clever-by-half, we explicitly force all inputs to be floats
    u = np.float64(u)
    v = np.float64(v)
    s = np.float64(s)
    a = np.float64(a)
    b = np.float64(b)
    rho = np.float64(rho)

    return u, s, v, a, b, rho


def naive_rank_one_update(u, s, v, a, b, rho):
    u, s, v, a, b, rho = sanity_check(u, s, v, a, b, rho)  # Make sure that inputs are good
    a_new = u @ s @ v.T + rho * np.outer(a, b)             # Perform rank-one update directly
    u, s, v = np.linalg.svd(a_new)                         # Decompose the updated matrix using Numpy
    return u, s, v


# Compute singular value decomposition of a rank-1 perturbation of a matrix with known singular value decomposition.
# Given a matrix with known singular value decomposition, A = U*E*V', the update efficiently calculates
# the SVD of the perturbed matrix, Ap = (A + rho*a*b'), where rho is some scalar and a and b are vectors.
#
# For details on computing eigen-vectors, see
#     "On the Efficient Update of the Singular Value Decomposition Subject to Rank-One Modifications", P. Stange, Proc.
#     Appl. Math. Mech.,8: 10827–10828. doi:10.1002/pamm.200810827
def rational_rank_one_update(u, s, v, a, b, rho, tol=1e-12):
    # Make sure that inputs are good
    u, s, v, a, b, rho = sanity_check(u, s, v, a, b, rho)
    (m, n) = s.shape

    # Initialize
    start_time = timer()    # Start the timer
    swapped = False         # Swap only if row dimension > column dimension
    a = rho * a             # Absorb rho into a

    # Get the dimensions of the matrix A or S. If N > M, then solve the transposed problem, and then transpose the
    # solution at the end. This can be very inefficient if called repeatedly
    if n > m:
        warnings.warn('\nSolving transposed problem! If you''re planning on calling this function multiple times, \ '
                      'consider transposing the problem setting for efficiency.', RuntimeWarning, stacklevel=2)
        swapped = True
        s = s.T         # Transpose the singular value matrix
        u, v = v, u     # Swap the right and left singular vectors
        a, b = b, a     # Transpose the rank-one update ab' to ba'
        m, n = n, m     # Swap the row and column dimensions

    # RANK-TWO update of V (left singular vectors) and E = S'S
    e0 = np.diag(s.T @ s)             # Eigenvalues s'*s
    a_tilde = v @ s.T @ u.T @ a       # Compute ai and bi
    x = np.array([[a.T @ a, 1], [1, 0]])
    r, q = schur(x)  # TODO: function to compute the Schur decomposition
    w = np.array([b, a_tilde]).T @ q
    ai = w[:, 0]
    bi = w[:, 1]

    e1, v1, diagnostic_e1 = eig_update(v, e0, v.T @ ai, r[0, 0], tol)
    e2, v2, diagnostic_e2 = eig_update(v1, e1, v1.T @ bi, r[1, 1], tol)

    # RANK-TWO update of U (right singular vectors) and D = SS'
    d0 = s @ s.T                      # Eigenvalues of s*s'
    b_tilde = u @ s @ v.T @ b         # Compute ao and bo
    x = np.array([[b.T @ b, 1], [1, 0]])
    r, q = schur(x)
    w = np.array([a, b_tilde]).T @ q
    ao = w[:, 0]
    bo = w[:, 1]

    d1, u1, diagnostic_d1 = eig_update(u, d0, u.T @ ao, r[0, 0], tol)
    d2, u2, diagnostic_d2 = eig_update(u1, d1, u1.T @ bo, r[1, 1], tol)

    # Reconstruct the singular value decomposition from the various RANK-2 updates above
    u_new = np.fliplr(u2)
    v_new = np.fliplr(v2)
    if m > n:
        s_new = np.concatenate([np.diag(np.sqrt(np.flipud(e2))), np.zeros((m - n, n))])
    else:
        s_new = np.concatenate([np.diag(np.sqrt(np.flipud(e2))), np.zeros((m, n - m))])

    # Compute the sign and rotate V appropriately
    sign = u_new.T @ u @ (s + u.T @ np.outer(a, b) @ v) @ v.T @ v_new
    tol = n * tol * np.max(np.diag(s_new))
    sign[np.abs(sign) < tol] = 0
    sign = np.sign(np.diag(sign))
    sign[sign == 0] = 1
    v_new = v_new @ np.diag(sign)

    # Undo the transpose that we might have done at the beginning if N > M
    if swapped:
        s_new = s_new.T
        u_new, v_new = v_new, u_new

    # Collect status and execution results into a diagnostic
    end_time = timer()
    diagnostic = {'total_time': end_time - start_time,
                  'tolerance': tol,
                  'eig_status': [diagnostic_e1, diagnostic_e2, diagnostic_d1, diagnostic_d2]}

    return u_new, s_new, v_new, diagnostic
