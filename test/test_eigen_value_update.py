import unittest
import numpy as np
from optimization.eigen_values import rational_rank_one_update
# import scipy.io as sio


class RankOneUpdateTest(unittest.TestCase):

    # Performs a simple test on a pre-defined problem
    def test_simple(self):
        v = np.identity(8)
        e = np.array([0, 0, 0, 5, 4, 3, 2, 1])

        u = np.array([8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625]).T
        rho = 1
        t = v.T @ u

        e_test, v_test, diagnostic = rational_rank_one_update(v, e, t, rho, tol=1e-12)

        try:
            # Ensure that the reconstruction error is within tolerance
            a_true = v @ np.diag(e) @ v.T + rho * np.outer(u, u)
            a_recon = v_test @ np.diag(e_test) @ v_test.T

            np.testing.assert_almost_equal(np.linalg.norm(a_true - a_recon, ord=2), 0.0, decimal=8)
            result = True
        except AssertionError as err:
            result = False
            print(err)
        self.assertTrue(result)

    # Performs test with several random eigenvalue decomposition with random rank-one updates; 25% of the matrices will
    # be positive definite (no zero eigenvalues) while the remaining will be sparse with between 30% to 40% zeros
    def test_random(self):
        for i in range(0, 50):  # Run this test several times
            n = np.random.randint(500, 751)
            e, v = self.generate_random_symmetric_matrix(n)  # Generate a random eigen-value decomposition

            if np.random.rand(1) > 0.25:
                # For some of the matrices, introduce sparsity in the eigen-values
                m = np.random.randint(np.floor(0.3 * n), np.ceil(0.45 * n))
                e[0:m] = 0.0
            else:
                m = 0

            print('Testing random matrix #{0} ({1} x {1}, rank = {2})...'.format(i, n, n - m), end='')

            u = np.random.randn(n)                        # Generate a random rank-one update vector
            t = v.T @ u                                      # Compute t = V'*u
            rho = np.random.randn(1)                         # Generate a random rho

            # Compute the update using Sylvanas
            # sio.savemat('CurrentMatrix.mat', {'v': v, 'e': e, 't': t, 'rho': rho})
            e_test, v_test, diagnostic = rational_rank_one_update(v, e, t, rho, tol=1e-12)

            try:
                # Ensure that the reconstruction error is within tolerance
                a_true = v @ np.diag(e) @ v.T + rho * np.outer(u, u)
                a_recon = v_test @ np.diag(e_test) @ v_test.T
                recon_err = np.linalg.norm(a_true - a_recon, ord=2)  # / (n*n)
                np.testing.assert_almost_equal(recon_err, 0.0, decimal=7)

                print(' reconstruction error = {0}.'.format(recon_err))
                result = True
            except AssertionError as err:
                result = False
                print(err)
                # print(' Failure variables written to file.\n')
                # sio.savemat('FailureReport.mat', {'v': v, 'e': e, 't': t, 'rho': rho})
            self.assertTrue(result)

    @staticmethod
    def generate_random_symmetric_matrix(n):
        a = np.random.randn(n, n)  # Generate a random (asymmetric) matrix of size n
        a = a.T @ a                # Compute A'A to make symmetric
        a = (a + a.T) / 2          # Make symmetry numerically stable
        e, v = np.linalg.eigh(a)   # Compute the eigen-decomposition of A
        return e, v


if __name__ == '__main__':
    unittest.main()
