import hashlib
import inspect
import numpy as np
import os
from learn import train
import tensorflow as tf

# take a function and return the hash value of its compiled bytecode
def sha224_hex(func):
    code = compile(inspect.getsource(func), '<string>', 'exec')
    return hashlib.sha224(code.co_code).hexdigest()

def smoothen(output_size, cache_dir='cached-nets', hash_func=sha224_hex, **ranges):
    """ Decorator that replaces a function with a "smoothed" neural network.

    This function will call into train.py to train a neural network on the
    functions's declared input ranges which will take some time if a cached
    network does not exist. All of the function's inputs and outputs should be
    numeric.

    Args:
        output_size: number of outputs of the function.
        cache_dir: location to look for and save trained neural nets.
        hash_func: hash function to use to hash function's bytecode. We use
            this to identify a trained neural net corresponding to a function.
        **ranges: Infinite iterators corresponding to each input of the function
            that generates values in the range of each input (see
            utils.iterators for examples).
    """
    def smoothed(func):
        # check if network for func has been cached
        fname = func.__name__ + '_' + hash_func(func)
        fname = os.path.join(cache_dir, fname)
        if os.path.isfile(fname):
            # TODO: implement saving/loading functionality.
            raise RuntimeError("saving/loading logic not implemented yet!")
            # net = train.load(fname)
        else:
            # if not, compute
            args, _, _, _ = inspect.getargspec(func)
            # will raise KeyError if there's an arg with an unspecified range.
            # this is intended.
            feature_iters = [ranges[arg] for arg in args]
            trainer = train.Trainer(
                    feature_iters=feature_iters,
                    output_size=output_size,
                    label=func)

            #trainer.train()
            trainer.test()
            #net = trainer.getNN()
            # save network
            # train.save(net, fname)
            return lambda *args: train.forward_pass(
                    net, np.array([args],dtype='float32'), trainer.x)
    return smoothed

