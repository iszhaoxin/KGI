import numpy as np
import sys, os
import chainer
from chainer import function_node
from chainer import reporter, Variable, Chain, link
from chainer import datasets, iterators, optimizers, training
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import chainer.computational_graph as C
from chainer import Function
import functools
import operator

from chainer.functions.connection import linear
from chainer import initializers
from chainer import variable


class LinearFunction(linear.LinearFunction):
    def forward(self, inputs):
        x, W, b = inputs
        y = x.dot(W.T).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        self.retain_inputs((0, 1))  # b is not retained
        return y,

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs
        ret = []
        with chainer.using_config('use_ideep', self._config_use_ideep):
            if 0 in indexes:
                gx, = linear.LinearGradData().apply((W, gy))
                ret.append(chainer.functions.cast(gx, x.dtype))
            if 1 in indexes:
                gW, = linear.LinearGradWeight(W.dtype).apply((x, gy))
                ret.append(chainer.functions.cast(gW, W.dtype))
            if 2 in indexes:
                gb = chainer.functions.sum(gy, axis=0)
                ret.append(gb)

        vdx = int(gW.shape[0]/2)
        hdx = int(gW.shape[1]/2)
        z = np.zeros((vdx, hdx), dtype=np.float32)
        gW.data[0:vdx, 0:hdx] = z
        gW.data[vdx:, hdx:] = z
        return ret

class LinearFunction2(linear.LinearFunction):
    def forward(self, inputs):
        x, W, b = inputs
        y = x.dot(W.T).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        self.retain_inputs((0, 1))  # b is not retained
        return y,

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs
        ret = []
        with chainer.using_config('use_ideep', self._config_use_ideep):
            if 0 in indexes:
                gx, = linear.LinearGradData().apply((W, gy))
                ret.append(chainer.functions.cast(gx, x.dtype))
            if 1 in indexes:
                gW, = linear.LinearGradWeight(W.dtype).apply((x, gy))
                ret.append(chainer.functions.cast(gW, W.dtype))
            if 2 in indexes:
                gb = chainer.functions.sum(gy, axis=0)
                ret.append(gb)

        vdx = int(gW.shape[0]/2)
        hdx = int(gW.shape[1]/2)
        z = np.zeros((vdx, hdx), dtype=np.float32)
        gW.data[0:vdx, hdx:] = z
        gW.data[vdx:, 0:hdx] = z
        return ret


def linearl(x, W, b=None, n_batch_axes=1):
    if n_batch_axes <= 0:
        raise ValueError('n_batch_axes should be greater than 0.')
    if n_batch_axes > 1:
        batch_shape = x.shape[:n_batch_axes]
        batch_size = np.prod(batch_shape)
        x = x.reshape(batch_size, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if b is None:
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction().apply(args)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1,))
    return y

def linearr(x, W, b=None, n_batch_axes=1):
    if n_batch_axes <= 0:
        raise ValueError('n_batch_axes should be greater than 0.')
    if n_batch_axes > 1:
        batch_shape = x.shape[:n_batch_axes]
        batch_size = np.prod(batch_shape)
        x = x.reshape(batch_size, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if b is None:
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction2().apply(args)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1,))
    return y

class MaskLinear(link.Link):
    def __init__(self, in_size, out_size=None, direction='l', nobias=False,
                 initialW=None, initial_bias=None):
        super(MaskLinear, self).__init__()
        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size
        self.direction = direction
        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size):
        self.W.initialize((self.out_size, in_size))

    def forward(self, x, n_batch_axes=1):
        if self.direction == 'l':
            return linearl(x, self.W, self.b, n_batch_axes=n_batch_axes)
        else:
            return linearr(x, self.W, self.b, n_batch_axes=n_batch_axes)

if __name__ == "__main__":
    batchsize = 12
    inputsize = 10
    outputsize = 6
    x = Variable(np.random.uniform(0, 1, (batchsize, inputsize)).astype(np.float32))
    W = Variable(np.random.uniform(0, 1, (outputsize, inputsize)).astype(np.float32))
    b = Variable(np.random.uniform(0, 1, (outputsize,)).astype(np.float32))
    
    loss = Variable(np.random.uniform(0, 1, (batchsize, outputsize)).astype(np.float32))
    l = LinearFunction()
    l.apply((x,W,b))
    l.backward((x,W,b), (loss))
    
    