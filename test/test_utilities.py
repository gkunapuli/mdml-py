import unittest
import numpy as np
import algorithms.utilities as util


class RankOneUpdateTest(unittest.TestCase):

    def test_asymmetric_triplet_generation(self):
        n_source = 10
        n_target = 15
        xs, ys, xt, yt = self.generate_two_domain_data(n_source, n_target)
        example_pairs, similarity_labels = util.generate_cross_domain_triples(xs, ys, xt, yt)

        try:
            # Ensure that the labels have been generated correctly
            test_labels = [2 * np.int(ys[pair[0]] == yt[pair[1]]) - 1 for pair in example_pairs]
            np.testing.assert_array_equal(test_labels, similarity_labels)
            result = True
        except AssertionError as err:
            result = False
            print(err)
        self.assertTrue(result)

    @staticmethod
    def generate_two_domain_data(n_source, n_target):
        xs = np.random.randn(n_source, 10)
        ys = 2 * (np.random.rand(n_source) > 0.5).astype(int) - 1
        xt = np.random.randn(n_target, 10)
        yt = 2 * (np.random.rand(n_target) > 0.5).astype(int) - 1

        return xs, ys, xt, yt


if __name__ == '__main__':
    unittest.main()
