"""
Generate Mel-frequency projection matrix.

Algorithm copied from Dan Ellis' rastamat toolbox for MATLAB.

Main symbol of interest here: `MelHtk`

"""
import functools
import numpy

def warp_dense(nfft, fft_max_freq, nfilts, 
    width=1.0,
    minfreq=0.0,
    maxfreq=None,
    const_amp=True,
    from_hz=None,
    to_hz=None):
    """
    :param nfft: Number of FFT bins spanning frequency 0 to double the nyquist.

    :param fft_max_freq: Highest frequency present in the Fourier transform (in Hz)

    :param nfilts: How many critical band amplitudes to extract from the power spectrum?

    :param width: Usually filters are set to half-overlap, corresponding to 1.0 here.

    :param minfreq: What is the lowest frequency to retrieve from the signal (lower edge of
    lowest bin, in Hz)

    :param maxfreq: What is the maximum frequency to retrieve from the signal (upper edge
    of highest bin, in Hz)

    :param const_amp: Should the projection filters have constant amplitude?   False gives
    them something more like constant energy.
    """

    wts = numpy.zeros((nfilts, nfft))# dtype='float64')
    fft_max_freq = float(fft_max_freq)
    width = float(width)
    minfreq = float(minfreq)
    maxfreq = float(fft_max_freq if maxfreq is None else maxfreq)

    #center freqs of each fft bin
    fftfreqs = numpy.asarray([(float(i)/nfft * fft_max_freq) for i in xrange(nfft)])

    #center freqs of mel bands - uniformly spaced between limits
    minmel = from_hz(minfreq)
    maxmel = from_hz(maxfreq)
    binfrqs = [to_hz(float(i)/(nfilts+1)*(maxmel-minmel)+minmel) 
            for i in range(nfilts+2)]

    #print binfrqs, maxfreq, to_hz(from_hz(maxfreq))

    for i in xrange(nfilts):

        # filter edges
        fs = numpy.asarray(binfrqs[i:i+3])

        #scale by width
        fs = fs[1]+width * (fs - fs[1])

        # lower and upper slopes for all bins
        loslope = (fftfreqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fftfreqs) / (fs[2] - fs[1])
        lolarger = loslope > hislope
        filt = lolarger * hislope + (1-lolarger) * loslope
        #print ''
        #print filt.shape, filt[:5], filt[-5:]
        #print fs
        filt = (filt > 0.0) * filt + (filt < 0) * 0.0
        #print filt[:20]

        if const_amp:
            wts[i,:] = filt
        else:
            #band width of this filter:
            bandwidth = (binfrqs[i+2]-binfrqs[i])
            wts[i,:] = filt * 2.0 / bandwidth

    return wts


def htk_mel_to_hz(w):
    return 700. * (pow(10., (w/2595.0))-1.)

def htk_mel_from_hz(f):
    return 2595. * numpy.log10(1. + f/700.0)

def MelHtk_warp_dense(*args, **kwargs):
    return warp_dense(
            to_hz=htk_mel_to_hz,
            from_hz=htk_mel_from_hz,
            *args, **kwargs)

def melhtk_4k(*args, **kwargs):
    return MelHtk_warp_dense(maxfreq=4000.0, *args, **kwargs)

def MelSlaney_warp_dense(*args, **kwargs):
    """args and kwargs passed to warp_dense
    """
    def from_hz(f):
        """:param f: frequency in hz"""
        f_0 = 0.0;
        f_sp = 200.0/3;
        brkfrq = 1000;
        brkpt = (brkfrq - f_0) / f_sp #starting mel for log region
        logstep = exp(log(6.4)/27);
        if f < brkfrq:
            return (f - f_0) / f_sp
        else:
            return brkpt + log(f / brkfrq)/log(logstep)
    
    def to_hz(w):
        raise NotImplementedError()

    return warp_dense(
            to_hz=to_hz,
            from_hz=from_hz,
            *args, **kwargs)

def MelMasters_warp_dense(nfft, fft_max_freq, nfilts, 
    width=1.0,
    minfreq=0.0,
    maxfreq=None,
    const_amp=True,
    to_hz=htk_mel_to_hz,
    from_hz=htk_mel_from_hz):
    """
    :param nfft: Number of FFT bins spanning frequency 0 to double the nyquist.

    :param fft_max_freq: Highest frequency present in the Fourier transform (in Hz)

    :param nfilts: How many critical band amplitudes to extract from the power spectrum?

    :param width: Usually filters are set to half-overlap, corresponding to 1.0 here.

    :param minfreq: What is the lowest frequency to retrieve from the signal (lower edge of
    lowest bin, in Hz)

    :param maxfreq: What is the maximum frequency to retrieve from the signal (upper edge
    of highest bin, in Hz)

    :param const_amp: Should the projection filters have constant amplitude?   False gives
    them something more like constant energy.
    """

    """
    Build a warp matrix corresponding to Htk's Mel-frequency formula, using `warp_dense`.

    In my masters I implemented this algorithm.

        matrix mel_warp()
        hz_nyquist = 11025,
        hz_spacing = 21,
        hz_maxidx = 512                                                                                                                                               
        mel_nyquist = mel(hz_nyquist),        mel_maxidx = 32                                                                                                                                               
                                                                                                                                                                                        
        W = zeros(hz_maxidx,mel_maxidx)                                                                                                                                                                     
        for (h = 0; h < hz_maxidx; ++h)                                                                                                                                                                     
        {                                                                                                                                                                                                   
        m_idx = mel( h * hz_spacing ) / mel_nyquist * mel_maxidx                                                                                                                                        
        j = floor(m_idx)                                                                                                                                                                                
        W[ h, j + 1] =       ( m_idx - j )                                                                                                                                                              
        W[ h, j + 0] = 1.0 - ( m_idx - j )                                                                                                                                                              
        }                                                                                                                                                                                                   
        normalize_col_sums( W, 1.0 ), return W               
    """

    wts = numpy.zeros((nfilts, nfft))# dtype='float64')
    fft_max_freq = float(fft_max_freq)
    assert width == 1.0

    minfreq = float(minfreq)
    maxfreq = float(fft_max_freq if maxfreq is None else maxfreq)

    minmel = from_hz(minfreq)
    maxmel = from_hz(maxfreq)

    hz_spacing = float(fft_max_freq) / nfft

    for h in xrange(nfft):
        # decide if this is a frequency bin we care about.
        h_in_hz = h * hz_spacing
        if (h_in_hz >= minfreq) and (h_in_hz <= maxfreq):

            # figure out which mel bin this power is headed to
            h_in_mel = from_hz(h_in_hz)
            mel_relative = (h_in_mel - minmel) / maxmel
            assert (mel_relative >= 0.0) and (mel_relative <= 1.0)
            mel_idx = mel_relative * nfilts

            m = int(mel_idx)

            if m+1 < nfilts:
                wts[m+1, h] = (mel_idx - m)
            wts[m+0, h] = 1.0 - (mel_idx - m)

    # normalize so that each Mel filter retrieves the same amount of power
    wts_T = wts.T
    wts_T *= 2.0 / (numpy.sum(wts, axis=1) * nfft)
    #print numpy.sum(wts, axis=1), numpy.sum(wts, axis=1).shape

    return wts

