# MDML-PY
## Mirror Descent for Metric Learning in Python

This project contains the Python implementation of MDML. **MDML** is **Mirror Descent for Metric Learning**, a unified approach to Mahalanobis metric learning. MDML is an online regularized metric learning algorithm based on the ideas of composite objective mirror descent ([COMID](https://web.stanford.edu/~jduchi/projects/DuchiShSiTe10.pdf)). The metric learning problem is formulated as a regularized positive semidefinite matrix learning problem, whose update rules can be derived using the COMID framework. This approach aims to be scalable, kernelizable, and admissible to many different types of Bregman and loss functions, which allows for the tailoring of several different classes of algorithms. MDML also uses the trace norm, which yields a sparse metric in its eigenspectrum, thus simultaneously performing feature selection along with metric learning.

MDML is more fully described in [this publication](https://gkunapuli.github.io/publication/12mdmlECML):<br>
G. Kunapuli and J. W. Shavlik. **Mirror Descent for Metric Learning: A Unified Approach**. _Twenty-Third European Conference on Machine Learning_ (ECML'12), Bristol, United Kingdom, September 24-29, 2012.

**Note**: The Python implementation is still in the alpha. For the (more robust) code used in the experiments from the ECML paper above, check out the [MATLAB version](https://github.com/gkunapuli/mdml) of MDML.
