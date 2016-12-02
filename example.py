# Just an example function demonstrating the use of the smooth decorator

import smooth
import random
#from utils.iterators import Continuous

@smooth.smoothen(
        output_size=1,
        a = random.randint(1,10),#Continuous(1,10),
        b = random.randint(1,10),#Continuous(1,10),
        c = random.randint(1,10))#Continuous(1,10))
def average(a, b, c):
    return round((a + b + c)/3.0)