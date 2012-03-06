""" Until Theano has a scan function,
these ops implement a scan of th covariance function over a 3D tensor
"""
import numpy, scipy.sparse
import theano

class CustomCovGrad(theano.Op):
    def make_node(self, x, g_c):
        return theano.Apply(self, [x, g_c], [x.type()])
    def perform(self, node, (x,g_c), (g_x,)):
        g_x[0] = numpy.zeros_like(x)
        for i, m  in enumerate(x):
            g_x[0][i] = (2.0/ m.shape[0]) * numpy.dot(m, g_c[i]) 
custom_cov_grad = CustomCovGrad()

class CustomCov(theano.Op):
    def make_node(self, x):
        x_ = theano.tensor.as_tensor_variable(x)
        if x_.ndim != 3:
            raise TypeError('need 3D tensor')
        bcastable = list(x_.broadcastable)
        out_type = theano.tensor.Tensor(dtype='float64',
                broadcastable=(bcastable[0],bcastable[2], bcastable[2]))
        return theano.Apply(self, [x_], [out_type()])
    def perform(self, node, (x,), (c,)):
        c[0] = numpy.empty((x.shape[0], x.shape[2], x.shape[2]), dtype='float64')
        for i, m  in enumerate(x):
            c[0][i] = numpy.dot(m.T, m) / m.shape[0]
    def grad(self, (x,), (g_c,)):
        return [custom_cov_grad(x, g_c)]

_custom_cov = CustomCov()

def custom_cov(x):
    means = theano.tensor.mean(x, axis=1)
    centered = x - means.dimshuffle(0,'x',1)
    return _custom_cov(centered)


class ScanOuterGrad(theano.Op):
    def make_node(self, x, g_c):
        return theano.Apply(self, [x,g_c], [x.type()])
    def perform(self, node, (x, g_c), (g_x,)):
        g_x[0] = numpy.empty(x.shape)
        for i, x_i  in enumerate(x):
            g_x[0][i] = numpy.dot(g_c[i], x_i) + numpy.dot(g_c[i].T, x_i)
scan_outer_grad = ScanOuterGrad()

class ScanOuter(theano.Op):
    def make_node(self, x):
        x_ = theano.tensor.as_tensor_variable(x)
        bcastable = list(x_.broadcastable)
        assert x_.ndim == 2
        assert x_.dtype == 'float64'
        out_type = theano.tensor.Tensor(dtype='float64', broadcastable=[False,False,False])
        return theano.Apply(self, [x_], [out_type()])
    def perform(self, node, (x,), (c,)):
        c[0] = numpy.empty((x.shape[0], x.shape[1], x.shape[1]), dtype='float64')
        for i, m  in enumerate(x):
            c[0][i] = numpy.outer(m, m)
    def grad(self, (x,), (g_c,)):
        return [scan_outer_grad(x, g_c)]
_scan_outer = ScanOuter()

def custom_corr(x, eps=1.0e-8):
    means = theano.tensor.mean(x, axis=1)
    vars = theano.tensor.var(x, axis=1) + eps
    stds = theano.tensor.sqrt(vars)
    centered = x - means.dimshuffle(0, 'x', 1)
    return custom_cov(centered) / _scan_outer(stds)



def pack_symmatrix_spmat(N, uplo='up'):
    """Return a sparse matrix for packing a flattened symmetic matrix into a shorter vector.
    
    Left-multiply with this sparse matrix to pack half of an NxN matrix (including the
    diagonal) that has been reshaped to a vector of length N*N into a vector of length
    N*(N+1)/2.

    With uplo=='up', iteration over the input matrix will begin by going over the top row, then
                     the second row starting at the diagonal, etc.
    With uplo=='lo', iteration will begin with elements [0,0],[0,1], [1,1], [0,2], [1,2]...
    """

    lil = scipy.sparse.lil_matrix((N*N, N*(N+1)/2), dtype='float64')
    if uplo == 'up':
        i_stride = N
        j_stride = 1
    elif uplo == 'lo':
        i_stride = 1
        j_stride = N
    else:
        raise NotImplementedError()
    k = 0
    for i in xrange(N): #
        for j in xrange(i, N):
            lil[i*i_stride + j*j_stride, k] = 1
            k += 1
    return lil.tocsc()

