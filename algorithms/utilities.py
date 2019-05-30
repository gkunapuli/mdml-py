from numpy.random import permutation
from numpy import concatenate, ones
from itertools import product, combinations


def generate_cross_domain_triples(xs, ys, xt, yt, n_similar=10000, n_dissimilar=20000):
    (n_source, d_source) = xs.shape
    (n_target, d_target) = xt.shape

    # Generate (source, target) example pairs via Cartesian product
    sim_triples = permutation([(i, j) for (i, j) in product(range(0, n_source), range(0, n_target)) if ys[i] == yt[j]])
    dis_triples = permutation([(i, j) for (i, j) in product(range(0, n_source), range(0, n_target)) if ys[i] != yt[j]])

    if len(sim_triples) < n_similar:
        n_similar = len(sim_triples)
    if len(dis_triples) < n_dissimilar:
        n_dissimilar = len(dis_triples)

    labels = concatenate([ones((n_similar, 1)), -ones((n_dissimilar, 1))])
    pairs = concatenate([sim_triples[:n_similar, :], dis_triples[:n_dissimilar, :]])
    triples = permutation(concatenate([pairs, labels], axis=1)).astype(int)

    return triples[:, 0:2], triples[:, 2]


def generate_within_domain_triples(x, y, n_similar=10000, n_dissimilar=20000):
    (n, d) = x.shape

    # Generate (ex_i, ex_j) example pairs via combinations
    sim_triples = permutation([(i, j) for (i, j) in combinations(range(0, n), 2) if y[i] == y[j]])
    dis_triples = permutation([(i, j) for (i, j) in combinations(range(0, n), 2) if y[i] != y[j]])

    if len(sim_triples) < n_similar:
        n_similar = len(sim_triples)
    if len(dis_triples) < n_dissimilar:
        n_dissimilar = len(dis_triples)

    labels = concatenate([ones((n_similar, 1)), -ones((n_dissimilar, 1))])
    pairs = concatenate([sim_triples[:n_similar, :], dis_triples[:n_dissimilar, :]])
    triples = permutation(concatenate([pairs, labels], axis=1).astype(int))

    return triples[:, 0:2], triples[:, 2]
