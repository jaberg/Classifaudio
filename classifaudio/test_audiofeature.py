
import wave #stdlib
import numpy

import theano
from audiofeature import AudioFeatures, mfcc_htk, mel_logspec

def load_blues_frames():
    w = wave.open('/u/bergstrj/tmp/blues.00000.wav')

    #this is the file I think it is right?
    assert w.getnchannels() == 1
    assert w.getsampwidth() == 2 #bytes -> 16bit
    assert w.getnframes() == 661794
    assert w.getframerate() == 22050

    sr = w.getframerate()
    framelen = 1024
    n_frames = (w.getnframes() // framelen)
    #print 'n_frames', n_frames

    samples = numpy.frombuffer(w.readframes(w.getnframes()), dtype='int16')
    samples = samples * 1.0 / (2**15)
    sample_max = samples.max()
    sample_min = samples.min()
    assert str(sample_max).startswith('0.88537')
    assert str(sample_min).startswith('-0.8402')

    w_ndarray = samples[:framelen * n_frames]
    assert w_ndarray.size == framelen * n_frames
    w_ndarray = w_ndarray.reshape(n_frames, framelen)

    return w_ndarray

def test_audiofeatures_from_wav():

    # Test that some functions exist and
    # at least run and return something

    w_ndarray = load_blues_frames()

    AF = AudioFeatures(sample_rate=22050, frame_len=1024)
    frames = theano.tensor.dmatrix('frames')
    assert AF.fft(frames).type == theano.tensor.zmatrix
    assert AF.spectrogram(frames).type == theano.tensor.dmatrix
    for name in ['fft', 'spectrogram', 'powspec', 'loudness', 'audspec', 'audcc']:

        fn = theano.function([frames], getattr(AF, name)(frames))
        rval = fn(w_ndarray)
        print name, rval.shape, rval.max(), rval.min(), rval.dtype

def test_powspec_vs_dan_ellis():
    # MATLAB:
    # [d, sr] = wavread('blues.00000.wav')
    # ref_spectrum = powspec(d, sr, .04645, .04645, 0);
    # size(ref_spectrum)
    # --> [513  646]
    # ref_spectrum(1:10, 1:5)
    ref_spectrum = 1.0e+11 * numpy.asarray([
        [0.0157,    0.0206,    0.0052,    0.00002,    0.0048],
        [0.0475,    0.0024,    0.0418,    0.0227,    0.0690],
        [0.2779,    0.2738,    0.1758,    0.3320,    0.0483],
        [0.0818,    0.7200,    1.2436,    1.1854,    1.2359],
        [0.0105,    0.0273,    0.2763,    0.1390,    0.6413],
        [0.0748,    0.8080,    1.5515,    1.2973,    1.7614],
        [0.0346,    3.5484,    2.9804,    3.0852,    0.0576],
        [0.0131,    0.7809,    0.0450,    0.1524,    0.9599],
        [0.0161,    0.5228,    0.4678,    0.3170,    0.5822],
        [0.1017,    0.6875,    0.4384,    0.3327,    0.7367]])


    M = theano.Module()
    M.audiofeatures = AudioFeatures(sample_rate=22050, frame_len=1024)
    frames = theano.tensor.dmatrix('frames')
    M.powspec = theano.Method([frames], M.audiofeatures.powspec_from_fft(
        fft=M.audiofeatures.fft(frames, half_fft=False, hamming=True),
        scale_to_int_range=True))
    m = theano.make_init(M)

    pspec = m.powspec(load_blues_frames()[:100,:])

    spectrum = pspec[0:5, 0:10].T

    print spectrum
    print abs(ref_spectrum - spectrum) / (ref_spectrum + spectrum)
    assert 0.01 > numpy.max( abs(ref_spectrum - spectrum) / (ref_spectrum + spectrum))

def test_audspec_vs_dan_ellis():
    # MATLAB:
    # [d, sr] = wavread('blues.00000.wav')
    # ref_spectrum = powspec(d, sr, .04645, .04645, 0);
    # size(ref_spectrum)
    # --> [513  646]
    # ref_audspec = audspec(ref_spectrum, sr, 12, 'htkmel');
    ref_audspec = 1.0e+11 * numpy.asarray([
        [0.4252,    5.8783,    6.5783,    5.2775,    4.9325],
        [0.3338,    1.0942,    1.6309,    0.9345,    1.2804],
        [0.0944,    0.1080,    0.2539,    0.1605,    0.7033],
        [0.0216,    0.0193,    0.3786,    0.2363,    0.4438],
        [0.1064,    0.0938,    0.6994,    0.9129,    0.5748],
        [0.2204,    0.2589,    2.2318,    0.9404,    0.7342],
        [0.0296,    0.0305,    0.2253,    0.2793,    0.2129],
        [0.0376,    0.0411,    0.1984,    0.1598,    0.0992],
        [0.0191,    0.0368,    0.2006,    0.0545,    0.0567],
        [0.0094,    0.0163,    0.1646,    0.0354,    0.0636]])



    M = theano.Module()
    M.audiofeatures = AudioFeatures(sample_rate=22050, frame_len=1024, n_bands=12)
    frames = theano.tensor.dmatrix('frames')
    power = M.audiofeatures.powspec_from_fft(
                M.audiofeatures.fft(frames, half_fft=True, hamming=True),
                scale_to_int_range=True)
    M.audspec = theano.Method([frames], 
        outputs=M.audiofeatures.audspec_from_power(power))
    m = theano.make_init(M)

    aspec = m.audspec(load_blues_frames()[:100,:])

    audspec = aspec[0:5, 0:10].T

    print audspec
    #print abs(ref_audspec - audspec) / (ref_audspec + audspec)
    max_rel_err = numpy.max( abs(ref_audspec - audspec) / (ref_audspec + audspec))
    assert 0.01 > max_rel_err


def test_mfcc_vs_dan_ellis():
    # MATLAB
    # ref_mfcc = melfcc(d, sr, 'wintime', 0.04645, 'hoptime', 0.04645, 'sumpower', 1, 'preemph', 0,
    #                          'lifterexp', 0, 'fbtype', 'htkmel');
    # ref_mfcc(:,1:4)
    ref_mfcc = numpy.asarray([
        [133.7590,  136.6686,  146.2690,  144.5163],
        [5.3952,    9.0799,    5.2408,    4.4615],
        [3.2458,    6.9730,    2.7541,    3.0766],
        [2.9110,    3.6644,    4.5135,    5.8142],
        [-1.7893,   -1.8694,    0.1458,   -0.1025],
        [-1.8752,   -2.1330,   -1.2386,   -0.6199],
        [1.7157,    1.3612,   -0.3314,   -0.1711],
        [0.8243,   -0.9234,   -1.5373,   -1.0007],
        [-1.4298,   -1.8253,   -1.1328,   -0.8016],
        [1.8595,    1.7053,    0.6195,    0.3924],
        [0.0345,   -0.7927,   -1.7412,   -0.9397],
        [0.4872,   -2.0002,    0.7030,   -0.1565],
        [1.5356,   -0.3391,   -0.2883,   -0.8777]])

    M = theano.Module()
    M.audiofeatures = AudioFeatures(sample_rate=22050, frame_len=1024, n_bands=40,
            critical_band_fn=mel.melhtk_4k)
    print M.audiofeatures.n_bands
    frames = theano.tensor.dmatrix('frames')
    power = M.audiofeatures.powspec_from_fft(
                M.audiofeatures.fft(frames, half_fft=True, hamming=True),
                scale_to_int_range=True)
    audspec = M.audiofeatures.audspec_from_power(power)
    print M.audiofeatures.n_bands
    M.mfcc = theano.Method([frames], 
        outputs=[M.audiofeatures.audcc_from_power(audspec, n_audcc=13), audspec])
    m = theano.make_init(M)

    blah, ablah = m.mfcc(load_blues_frames()[:100,:])
    assert ablah.shape == (100, 40)
    assert blah.shape == (100, 13)
    my_mfcc = blah[0:4,:13].T

    rel_err = numpy.max( abs(ref_mfcc - my_mfcc) / (abs(ref_mfcc) + abs(my_mfcc)+1.0e-8))
    if rel_err >= 0.01:
        print 'BLAH SHAPE', blah.shape
        print ablah[0:5,:13].T
        print my_mfcc
        print abs(ref_mfcc - my_mfcc) / (abs(ref_mfcc) + abs(my_mfcc) + 1.0e-8)
    assert rel_err < 0.01

def test_audiofeatures_fft():
    M = theano.Module()
    M.audiofeatures = AudioFeatures(sample_rate=22050, frame_len=1024)
    frames = theano.tensor.dmatrix('x')
    M.fft = theano.Method([frames], M.audiofeatures.fft(frames, half_fft=False))
    m = theano.make_init(M)

    rng = numpy.random.RandomState(4234)
    xval = rng.randn(100, 1024)
    assert numpy.allclose(m.fft(xval), numpy.fft.fft(xval, 1024, 1))

def test_segment():

    frames = theano.tensor.dmatrix('frames')
    f  = theano.function([frames], segment(frames, 5))

    val = numpy.random.RandomState(34).randn(20, 3)
    val2 = numpy.random.RandomState(34).randn(5, 4, 3)

    assert numpy.allclose(f(val), val2)



from pylearn.datasets import tzanetakis
from pylearn.io import wavread

def test_melraw_bel_60_9():
    target_file = "/u/bergstrj/cvs/bergstrj/articles/06_memoir/data/feat_melraw_bel_60_9.stat"
    file1_features = numpy.asarray([float(token) for token in file(target_file).readline().split()])
    file1_features = file1_features.reshape((9, 560))

    masters_means = file1_features[:, :32]
    masters_covars = file1_features[:, 32:]

    idx = theano.tensor.lscalar()
    path, label = tzanetakis.tzanetakis_example(idx)
    samples, sr = wavread.wav_read_double(path)
    frames = (samples[:1024*640]).reshape((640,1024))

    segment1_frames=frames[:60,:]
    AF = AudioFeatures(22050, 1024, 
            n_bands=32,
            scale_to_int_range=False,
            critical_band_fn=mel.MelMasters.warp_dense,
            power_sqr_abs=False,
            noise_level=1.0e-4) #1/10000
    
    AF.f = theano.Method([idx], [AF.loudness(segment1_frames), sr])

    af = AF.make()

    segment1_features, actual_sr = af.f(0)

    assert actual_sr == 22050

    print numpy.mean(segment1_features, axis=0)
    print masters_means[0]

    print numpy.mean(segment1_features, axis=0)    - masters_means[0]

