import tensorflow as tf
import numpy as np
import sys


def lookup(what, where, default=None):
    """Return ``where.what`` if what is a string, otherwise what. If not found
    return ``default``."""
    if sys.version_info <= (3,0):
        if isinstance(what, (str, unicode)):
            res = getattr(where, what, default)
        else:
            res = what
    else:
        if isinstance(what, (bytes, str)):
            res = getattr(where, what, default)
        else:
            res = what
    return res


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out)) 
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

    
class Layer(object):

    def __init__(self, name=None):
        pass

    def __call__(self):
        raise NotImplemented
        

class AffineNonlinear(Layer):

    def __init__(self, n_inpt, n_output, transfer=lambda x: x,
                 use_bias=True, name=None):
        self.n_inpt = n_inpt
        self.n_output = n_output
        self.transfer = transfer
        self.use_bias = use_bias

        self.weights = tf.Variable(xavier_init(self.n_inpt, self.n_output))
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32))

        super(AffineNonlinear, self).__init__(name=name)

    def __call__(self, inpt):
        output_in = tf.matmul(inpt, self.weights)

        if self.use_bias:
            output_in = tf.add(output_in, self.bias)

        f = lookup(self.transfer, tf.nn)

        return f(output_in)
        

class Mlp(Layer):

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, name=None):
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer

        self.layers = []
        self.n_inpts = [self.n_inpt] + self.n_hiddens
        self.n_outputs = self.n_hiddens + [self.n_output]
        self.transfers = self.hidden_transfers + [self.out_transfer]

        for n, m, t in zip(self.n_inpts, self.n_outputs, self.transfers):
            layer = AffineNonlinear(n, m, t)
            self.layers.append(layer)

        super(Mlp, self).__init__(name)

    def __call__(self, inpt):
        output = inpt
        for l in self.layers:
            output = l(output)

        return output