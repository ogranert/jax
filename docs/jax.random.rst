``jax.random`` module
=====================

.. automodule:: jax.random

API Reference
-------------

Key Creation & Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

  PRNGKey
  key
  key_data
  wrap_key_data
  fold_in
  split

Random Samplers
~~~~~~~~~~~~~~~

.. Generate the list of callable members:
   >>> from jax import random
   >>> fns = (x for x in sorted(dir(random)) if callable(getattr(random, x)))
   >>> print('\n'.join('    ' + x for x in fns))  # doctest: +SKIP

.. autosummary::
  :toctree: _autosummary

    ball
    bernoulli
    beta
    bits
    categorical
    cauchy
    chisquare
    choice
    dirichlet
    double_sided_maxwell
    exponential
    f
    gamma
    generalized_normal
    geometric
    gumbel
    laplace
    loggamma
    logistic
    lognormal
    maxwell
    multivariate_normal
    normal
    orthogonal
    pareto
    permutation
    poisson
    rademacher
    randint
    rayleigh
    shuffle
    t
    triangular
    truncated_normal
    uniform
    wald
    weibull_min
