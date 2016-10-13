# Some utility iterators for generating function inputs

import numpy as np

class Discrete:
    """ A random infinite iterator over a (finite) set of discrete values."""
    def __init__(self, *values):
        self.values = values

    def __next__(self):
        # returns a random element from values with replacement, so this
        # iterator is infinite
        return np.random.choice(self.values)

class Continuous:
    """ A random infinite iterator over a range of values. """
    # expects a single bound (possible extension if needed: accept multiple
    # tuples each representing a valid range).
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __next__(self):
        # returns a value in range [lo,hi)
        return np.random.random() * (self.hi - self.lo) + self.lo


