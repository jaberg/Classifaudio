"""Provides `AudioFeatures`
"""
#system imports
import numpy
import scipy.sparse


import theano
import theano.sparse
from theano.tensor import as_tensor, Op, Apply, Tensor
from theano.sandbox import fourier

#local imports
import mel

class AudioFeatures(object):
    """A Module for extracting audio features for music classification"""

    n_bands = 64
    """Number of critical bands to extract (for audspec)"""

    noise_level = 1.0
    """Amount to add to power before taking log 
    
    (should depend on `frame_len`)
    
    an amount of white noise to add to the signal (works around undefined-ness of the loudness
    of zero).  (WRITEME noise relative to what amplitude)
    """

    critical_band_fn = mel.MelHtk_warp_dense
    """Psychoacoustic critical band model (see `mel` module for alternatives)
    
    a function taking three arguments (n_fft, fft_max_freq, n_bands)  and returning a dense
    matrix that implements frequency-scaling of a power spectrum.  See `mel.MelHtk.warp_dense`
    as an example.
    """

    half_fft = True
    """When computing the FFT, skip the frequencies above Nyquist.
    """

    hamming = False
    """Apply a hamming window to each frame prior to FFT"""

    n_audcc = 13
    """How many Cepstral coefficients (e.g. MFCC) to extract from critical bands?"""

    scale_to_int_range = True
    """Should FFT responses be scaled by (2**15)?  

    You might consider doing this if post-processing was tuned for integer-valued audio
    samples, but the input consists of real-valued audio samples on the [-1,1] range.
    """

    power_sqr_abs = True
    """
    """

    use_sparse_warp = True

    dct_unitary = True

    def __init__(self, sample_rate, frame_len, **kwargs):
        """
        :param sample_rate: sample rate of audio in the frames

        :param frame_len: samples per audio frame (should be power of 2)

        Other arguments over-ride the class defaults.
        """
        super(AudioFeatures, self).__init__()
        self.sample_rate = sample_rate
        self.frame_len = frame_len   # number of samples per audio frame
        if int(frame_len) <= 0:
            raise TypeError('Frame len must be positive integer')

        # over-ride class defaults
        for attr in kwargs:
            if not hasattr(self, attr):
                raise TypeError('unknown keyword argument', attr)
            setattr(self, attr, kwargs[attr])
            assert getattr(self, attr) is kwargs[attr]

        self.params = [] # built on demand as features are requested

    def fft(self, frames, half_fft=None, hamming=None, scale_to_int_range=None):
        #(optional) apply hamming window 
        half_fft = self.half_fft if half_fft is None else half_fft
        hamming = self.hamming if hamming is None else hamming

        scale_to_int_range = self.scale_to_int_range \
            if scale_to_int_range is None else scale_to_int_range

        if hamming:
            _frames = frames * numpy.hamming(self.frame_len)
        else:
            _frames = frames

        if scale_to_int_range:
            _frames = _frames * (2**15)

        if half_fft:
            return fourier.half_fft(_frames, n=self.frame_len, axis=1)
        else:
            return fourier.fft(_frames, n=self.frame_len, axis=1)

    def spectrogram(self, frames):
        """Absolute value of FFT of each frame"""
        return abs(self.fft(frames))

    def powspec(self, frames, power_sqr_abs=None):
        """
        :param frames: each row a frame of [mono] audio. signal should be in [-1,1] range
        """
        power_sqr_abs = self.power_sqr_abs if power_sqr_abs is None else power_sqr_abs

        return self.powspec_from_fft(self.fft(frames), power_sqr_abs)

    def powspec_from_fft(self, fft, power_sqr_abs=None):
        power_sqr_abs = self.power_sqr_abs \
                if power_sqr_abs is None else power_sqr_abs

        return abs(fft)**2 if power_sqr_abs else abs(fft)

    def audspec(self, frames, use_sparse_warp=None):
        """
        :param frames: each row a frame of [mono] audio. signal should be in [-1,1] range

        Return the loudness of each critical band per frame.
        """
        return self.audspec_from_power(self.powspec(frames), use_sparse_warp=use_sparse_warp)

    def audspec_from_power(self, power, use_sparse_warp=None):
        use_sparse_warp = self.use_sparse_warp if use_sparse_warp is None else use_sparse_warp

        warp = self.critical_band_fn(
                nfft=self.frame_len/2,
                fft_max_freq=self.sample_rate/2,
                nfilts=self.n_bands)
        name=name="AudioFeature.warp<%i>"%id(warp)

        if use_sparse_warp:
            warp = theano.sparse.as_sparse_variable(scipy.sparse.csr_matrix(warp), name=name)
        else:
            warp = tensor.as_tensor_variable(warp, name=name)
        return theano.dot(power, warp.T)

    def audcc(self, frames, n_audcc=None):
        """Compute cepstral coefficients, such as MFCCs.

        :param frames: audio frames
        
        :param n_audcc: number of cepstral coefficients to calculate

        :returns: symbolic ndarray. Cepstral coefficients of each frame in each row.

        """
        n_audcc = self.n_audcc if n_audcc is None else n_audcc

        rval =  self.audcc_from_power(self.audspec(frames), n_audcc=n_audcc)
        rval.tag.shape = (None, n_audcc)
        return rval

    def audcc_from_power(self, power, n_bands=None, n_audcc=None, dct_unitary=None,
            noise_level=None):
        """
        :type power: ndarray or NdArrayResult with ndim=2

        :param power: a power spectrogram with each frame in a row.  A frequency-scaled
        spectrogram makes sense here too.

        :type n_bands: int
        :param n_bands:  number of critical bands of power

        :type n_audcc: int
        :param n_audcc:  number of cepstral coefficients to calculate

        :type dct_unitary: Bool
        :param dct_unitary: True means apply different scaling to first coef.

        """
        n_audcc = self.n_audcc if n_audcc is None else n_audcc
        dct_unitary = self.dct_unitary if dct_unitary is None else dct_unitary
        n_bands = self.n_bands if n_bands is None else n_bands
        noise_level = self.noise_level if noise_level is None else noise_level

        dct = fourier.dct_matrix(n_audcc, n_bands, unitary=dct_unitary)

        dct = theano.tensor.as_tensor_variable(dct, name="AudioFeatures.dct<%i>"%id(dct))
        return theano.dot(theano.tensor.log(power + noise_level), dct.T)

    def loudness(self, frames):
        return self.loudness_from_power(self.audspec(frames))

    def loudness_from_power(self, power, noise_level=None):
        noise_level = self.noise_level if noise_level is None else noise_level
        return theano.tensor.log10(power + noise_level)

def mfcc_htk(frames, n_mfcc, sr, wlen=None):
    """
    """
    if wlen is None:
        wlen = frames.tag.shape[1]
    AF = AudioFeatures(
            sample_rate=sr,
            frame_len=wlen,
            n_bands=2*n_mfcc+4,
            n_audcc=n_mfcc,
            half_fft=True,
            hamming=True,
            scale_to_int_range=True,
            critical_band_fn=mel.melhtk_4k,
            power_sqr_abs=True,
            noise_level=1)
    return AF.audcc(frames)

def mel_logspec(frames, n_coefs, sr, wlen=None):
    if wlen is None:
        wlen = frames.tag.shape[1]
    try:
        n_frames = frames.tag.shape[0]
    except AttributeError:
        n_frames = None

    AF = AudioFeatures(sr, wlen, 
            n_bands=n_coefs,
            scale_to_int_range=False,
            critical_band_fn=mel.MelMasters_warp_dense,
            power_sqr_abs=False,
            noise_level=1.0e-4) #1/10000

    output = AF.loudness(frames)+4.0001 # the 4 is from 1.e-4 of noise, the epsilon for rounding errs
    output.tag.shape = (n_frames, n_coefs)
    return output
