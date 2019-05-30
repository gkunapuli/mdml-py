import unittest
import numpy as np
from optimization.singular_values import rational_rank_one_update
# import scipy.io as sio


class RankOneUpdateTest(unittest.TestCase):

    # Performs a simple test on a pre-defined problem
    def test_simple(self):
        u = np.identity(8)
        v = np.identity(5)
        s = np.concatenate([np.diag([5, 4, 3, 2, 1]), np.zeros((3, 5))], axis=0)

        a = 2.0 ** np.arange(-2, 6)
        b = 3.0 ** np.arange(2, -3, -1)
        rho = 1.5

        u_test, s_test, v_test, diagnostic = rational_rank_one_update(u, s, v, a, b, rho, tol=1e-12)

        try:
            # Ensure that the reconstruction error is within tolerance
            a_true = u @ s @ v.T + rho * np.outer(a, b)
            a_recon = u_test @ s_test @ v_test.T

            np.testing.assert_almost_equal(np.linalg.norm(a_true - a_recon, ord=2), 0.0, decimal=8)
            result = True
        except AssertionError as err:
            result = False
            print(err)
        self.assertTrue(result)

    # Performs test with several random SVD with random rank-one updates; 25% of the matrices will be full rank (no zero
    #  singular values), while the remaining will be sparse with between 30% to 40% zero singular values
    def test_random(self):
        for i in range(0, 50):  # Run this test several times
            m = np.random.randint(100, 251)
            n = np.int(np.floor(m * np.clip(np.random.rand(), 0.4, 0.7)))
            u, s, v = self.generate_random_symmetric_matrix(m, n)  # Generate a random SVD

            if np.random.rand(1) > 0.25:
                # For some of the matrices, introduce sparsity in the eigen-values
                z = np.random.randint(np.floor(0.3 * n), np.ceil(0.45 * n))
                s[0:z, 0:z] = 0.0
            else:
                z = 0

            print('Testing random matrix #{0} ({1} x {2}, rank = {3})...'.format(i, m, n, n - z), end='')

            a = np.random.randn(m)   # Generate a random rank-one update vector
            b = np.random.randn(n)   # Generate a random rank-one update vector
            rho = np.random.randn(1) # Generate a random rho

            # Compute the update using Sylvanas
            # sio.savemat('CurrentMatrix.mat', {'v': v, 'e': e, 't': t, 'rho': rho})
            u_test, s_test, v_test, diagnostic = rational_rank_one_update(u, s, v, a, b, rho, tol=1e-12)

            try:
                # Ensure that the reconstruction error is within tolerance
                a_true = u @ s @ v.T + rho * np.outer(a, b)
                a_recon = u_test @ s_test @ v_test.T
                recon_err = np.linalg.norm(a_true - a_recon, ord=2) / (n*n)
                np.testing.assert_almost_equal(recon_err, 0.0, decimal=8)

                print(' reconstruction error = {0}.'.format(recon_err))
                result = True
            except AssertionError as err:
                result = False
                print(err)
                # print(' Failure variables written to file.\n')
                # sio.savemat('FailureReport.mat', {'v': v, 'e': e, 't': t, 'rho': rho})
            self.assertTrue(result)

    @staticmethod
    def generate_random_symmetric_matrix(m, n):
        a = np.random.randn(m, n)                        # Generate a random (asymmetric) matrix of size n
        u, s, v = np.linalg.svd(a, full_matrices=True)   # Compute the eigen-decomposition of A
        s = np.concatenate([np.diag(s), np.zeros((m - n, n))])
        return u, s, v


if __name__ == '__main__':
    unittest.main()
