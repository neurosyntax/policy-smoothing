# Just an example function demonstrating the use of the smooth decorator

import smooth
import random
from utils.iterators import Continuous

@smooth.smoothen(
        output_size=1,
        a = Continuous(1,10),
        b = Continuous(1,10),
        c = Continuous(1,10))
def average(a, b, c):
    return (a + b + c)/3.0