import sys
import numpy
import theano
import audiofeature, mel
from cov_op import custom_cov, custom_corr, pack_symmatrix_spmat

def segment_features(features, n_segments):
    """Reshape a vector of rows (2D array) into a matrix of rows (3D array)"""
    if features.ndim == 2:
        n_rows, n_cols = features.shape
        segments = theano.tensor.reshape(features, (n_segments, n_rows / n_segments, n_cols))
    else:
        raise NotImplementedError()
    return segments

def segment_mean(segments):
    #mean over time within each segment
    return theano.tensor.mean(segments, axis=1)

def segment_var(segments):
    #mean over time within each segment
    return theano.tensor.var(segments, axis=1)

def segment_corr(segments):
    #correlation over time within each segment
    n_segments = segments.shape[0]
    width = segments.shape[2]
    return custom_corr(segments).reshape((n_segments, width**2))

def segment_cov(segments):
    #cov over time within each segment
    n_segments = segments.shape[0]
    width = segments.shape[2]
    return custom_cov(segments).reshape((n_segments, width**2))

def segment_packed_cov(segments, width=None):
    #cov over time within each segment
    if width is None:
        width = segments.tag.shape[2]
    n_segments = segments.shape[0]
    packing_matrix = pack_symmatrix_spmat(width)
    cov = custom_cov(segments).reshape((n_segments, width**2))
    rval = theano.sparse.structured_dot(cov, packing_matrix)
    if cov.type.broadcastable[0]:
        # sparse matrices have no broadcastable property,
        # so the broadcastable[0] is always False, even when it shouldn't be
        rval = theano.tensor.Rebroadcast((0, True))(rval)
    return rval

def segment_packed_corr(segments, width=None):
    #correlation over time within each segment
    if width is None:
        width = segments.tag.shape[2]
    try:
        n_segments = segments.tag.shape[0]
    except AttributeError:
        n_segments = segments.shape[0]
    packing_matrix = pack_symmatrix_spmat(width)
    corr = custom_corr(segments).reshape((n_segments, width**2))
    rval = theano.sparse.structured_dot(corr, packing_matrix)
    if corr.type.broadcastable[0]:
        # sparse matrices have no broadcastable property,
        # so the broadcastable[0] is always False, even when it shouldn't be
        rval = theano.tensor.Rebroadcast((0, True))(rval)
    return rval

class SegmentStats(theano.Module):
    def __init__(self, features, features_shape1, n_segments, segment_fnlist):
        super(SegmentStats, self).__init__()
        self.features = features
        segments = segment_features(features, n_segments)
        segments_shape2 = features_shape1
        assert segments.ndim == 3
        self.segments = segments

        feature_stats = []
        n_features = 0
        for fn in segment_fnlist:
            stats, nstats = fn(segments, n_segments, segments_shape2)
            feature_stats.append(stats)
            n_features += nstats

        self.output = theano.tensor.join(1, *feature_stats)
        self.output_shape1 = n_features




#ORIGINAL FEATURE FUNCTIONS
###########################
###########################

def mfcc_features(frames, n_mfcc=16, n_segments=10):
        af = audiofeature.AudioFeatures(
                sample_rate=22050,
                frame_len=1024,
                n_bands=2*n_mfcc+4,
                n_audcc=n_mfcc,
                half_fft=True,
                hamming=True,
                scale_to_int_range=True,
                critical_band_fn=mel.melhtk_4k,
                power_sqr_abs=True,
                noise_level=1)
        mfcc = af.audcc(frames)
        mfcc_segments = audiofeature.segment(mfcc, n_segments)
        return af, mfcc_segments, n_mfcc

def mpc_features(frames, n_mpc=32, n_segments=10):
        AF = audiofeature.AudioFeatures(22050, 1024, 
                n_bands=n_mpc,
                scale_to_int_range=False,
                critical_band_fn=mel.MelMasters.warp_dense,
                power_sqr_abs=False,
                noise_level=1.0e-4) #1/10000

        mpc = AF.loudness(frames)
        mpc_segments = audiofeature.segment(mpc, n_segments)
        return AF, mpc_segments, n_mpc

def mpc_w_softplus(frames, n_mpc=32, n_segments=10):
        AF = audiofeature.AudioFeatures(22050, 1024, 
                n_bands=n_mpc,
                scale_to_int_range=False,
                critical_band_fn=mel.MelMasters.warp_dense,
                power_sqr_abs=False,
                noise_level=1.0e-4) #1/10000

        mpc = theano.tensor.log10(0.01 * theano.tensor.nnet.softplus(100*AF.audspec(frames)) + 1.0e-4)
        mpc_segments = audiofeature.segment(mpc, n_segments)
        return AF, mpc_segments, n_mpc

def mpc_w_clipping(frames,  n_mpc=32, n_segments=10, use_sparse_warp=False):
    AF = audiofeature.AudioFeatures(22050, 1024, 
            n_bands=n_mpc,
            scale_to_int_range=False,
            critical_band_fn=mel.MelMasters.warp_dense,
            power_sqr_abs=False,
            use_sparse_warp=use_sparse_warp,
            noise_level=1.0e-4) #1/10000

    audspec = AF.audspec(frames)
    mpc = theano.tensor.log10(theano.tensor.switch(audspec>AF.noise_level, audspec, AF.noise_level))
    mpc_segments = audiofeature.segment(mpc, n_segments)
    return AF, mpc_segments, n_mpc

def mpc_w_max_approx(frames, n_mpc=32, n_segments=10, use_sparse_warp=False):
        AF = audiofeature.AudioFeatures(22050, 1024, 
                n_bands=n_mpc,
                scale_to_int_range=False,
                critical_band_fn=mel.MelMasters.warp_dense,
                power_sqr_abs=False,
                noise_level=1.0e-4) #1/10000

        powspec = AF.powspec(frames) #abs(fft(x))
        logspec = theano.tensor.log10(powspec + AF.noise_level)
        logspec3 = theano.tensor.DimShuffle(logspec.broadcastable, [0, 'x', 1])(logspec)
        # warp_mat dims: nfilts x nfft
        warp_mat = AF.critical_band_fn(
                nfft=AF.frame_len/2,
                fft_max_freq=AF.sample_rate/2,
                nfilts=AF.n_bands) 
        log_warp_mat = numpy.log10(warp_mat + 1.0e-12)

        sum3 = theano.tensor.add(log_warp_mat,logspec3)
        feature = theano.tensor.max(sum3)
        mpc_segments = audiofeature.segment(feature, n_segments)
        return AF, mpc_segments, n_mpc


def mpc_w_max_approx_learnable(frames, n_mpc=32, n_segments=10, use_sparse_warp=False):
    def log_warp_mat_fn(*args, **kwargs):
        m = mel.MelMasters.warp_dense(*args, **kwargs)
        return numpy.log10(m + 1.0e-12)

    AF = audiofeature.AudioFeatures(22050, 1024, 
            n_bands=n_mpc,
            scale_to_int_range=False,
            critical_band_fn=log_warp_mat_fn,
            power_sqr_abs=False,
            noise_level=1.0e-4) #1/10000

    powspec = AF.powspec(frames) #abs(fft(x))
    logspec = theano.tensor.log10(powspec + AF.noise_level)
    logspec3 = theano.tensor.DimShuffle(logspec.broadcastable, [0, 'x', 1])(logspec)
    sum3 = theano.tensor.add(AF.critical_band_warp_dense, logspec3)
    feature = theano.tensor.max(sum3)
    mpc_segments = audiofeature.segment(feature, n_segments)
    return AF, mpc_segments, n_mpc

