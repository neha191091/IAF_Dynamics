import tensorflow as tf
import numpy as np
import sys
slim = tf.contrib.slim

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

# TODO: Check if correct

class AR_Layer(Layer):
    def __init__(self, mask, name=None):
        self.W = tf.Variable(tf.truncated_normal(mask.shape, stddev = 0.1))
        self.mask = mask
        self.b = tf.Variable(tf.constant(0., shape = [mask.shape[1], ]))
        super(AR_Layer, self).__init__(name)

    def __call__(self, x, applyActivation):
        W = tf.multiply(self.W, self.mask)
        output = tf.add(tf.matmul(x, W), self.b)
        if applyActivation is True:
            output = tf.nn.elu(output)
        return output


class AR_Net(Layer):
    def __init__(self, z_size, h_size, shape = [256, 256], name=None):
        self.masks = ar_masks(z_size, h_size, shape)
        self.z_size = z_size
        self.layers = []
        for i in range(len(shape)):
            hidden = AR_Layer(self.masks[i])
            self.layers.append(hidden)
        self.layers.append(AR_Layer(self.masks[len(shape)]))
        super(AR_Net, self).__init__(name)

    def __call__(self, h, z_init):
        output = tf.concat([h, z_init], 1)
        n_layers = len(self.layers)
        for i in range(n_layers):
            layer = self.layers[i]
            if i < n_layers-1:
                output = layer(output, True)
            else:
                output = layer(output, False)
        s = output[:, :self.z_size]
        mean = output[:, self.z_size:]
        var_f0 = tf.nn.sigmoid(s)
        #var_f0 = s ** 2
        #z_temp = mean + var_f0 * z_init
        z_temp = (1 - var_f0) * mean + var_f0 * z_init
        #z_temp = tf.reverse(z_temp, [False, True])
        return var_f0, z_temp


def ar_masks(z_size, h_size, shape = [256, 256]):
    masks = []

    n_in = h_size + z_size
    id_in = np.concatenate(([-1 for _ in range(h_size)], range(z_size)))

    for m in range(len(shape)):

        n_out = shape[m]
        id_out = np.random.randint(0, z_size - 1, size = n_out)
        mask = np.zeros([n_in, n_out], dtype = np.float32)
        for k_in in range(n_in):
            for k_out in range(n_out):
                if id_out[k_out] >= id_in[k_in]:
                    mask[k_in, k_out] = 1
        masks.append(mask)

        n_in = n_out
        id_in = id_out

    n_out = 2 * z_size
    id_out = np.concatenate((range(z_size), range(z_size)))
    mask = np.zeros([n_in, n_out], dtype = np.float32)
    for k_in in range(n_in):
        for k_out in range(n_out):
            if id_out[k_out] > id_in[k_in]:
                mask[k_in, k_out] = 1
    masks.append(mask)
    return masks