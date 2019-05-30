import unittest
import numpy as np
from algorithms.utilities import generate_within_domain_triples as generate_triples
from algorithms.mahalanobis_metric import learn
# import scipy.io as sio


class RankOneUpdateTest(unittest.TestCase):

    def test_2d_explicit_update(self):
        n = 100
        d = 10
        x, y, valid_dim = self.generate_2d_data(n, d)
        pairs, labels = generate_triples(x, y, n_similar=100, n_dissimilar=200)
        e, v, mu, total_time, num_updates = learn(x, pairs, labels,
                                                  eta=1.0, rho=2.0, loss='hinge', bregman_function='Frobenius',
                                                  tol=1e-12, algorithm='rational', verbose=True, debug=True)

        self.assertTrue(x.shape[0] == 100)

    # Generate 2-dimensional data with a number of examples (n) of dimension (d). The data includes garbage dimensions
    # (of size d-2).
    @staticmethod
    def generate_2d_data(n, d):
        M = np.array([0, 0])
        C = np.array([[2, 0.2], [0.2, 2]])

        x_valid = np.random.multivariate_normal(M, C, 10*n)
        x_norm = np.sqrt(np.sum(x_valid * x_valid, axis=1))

        idx1 = np.where(x_norm < 0.85)[0]
        idx2 = np.where(x_norm > 1.25)[0]
        x_valid = np.concatenate([x_valid[idx1[:n//2], :], x_valid[idx2[:n//2]]], axis=0)

        y = np.ones((n,))
        y[n//2+1:] += 1

        # Now add 8 additional garbage dimensions. A good metric learning approach that learns low-dimensional
        # representations should be able to strip these during learning
        x = np.random.randn(n, d)                 # Generate d-dimensional garbage data
        i_valid = np.random.permutation(10)[0:2]  # Pick two random dimensions to save the good data
        x[:, i_valid] = x_valid                   # Insert our data set into those dimensions

        return x, y, i_valid


if __name__ == '__main__':
    unittest.main()
