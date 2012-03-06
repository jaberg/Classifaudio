import numpy
import theano
import theano.tensor.nnet
from theano import tensor
import classifaudio

class HalflifeStopper(object):
    """An early-stopping crition.

    This object will track the progress of a dynamic quantity along some noisy U-shaped
    trajectory.

    The heuristic used is to first iterate at least `initial_wait` times, while looking at the
    score.  If at any point thereafter, the score hasn't made a *significant* improvement in the
    second half  of the entire run, the run is declared *not*-`promising`.

    Significant improvement in the second half of a run is defined as achieving
    `progresh_thresh` proportion of the best score from the first half of the run.

    Instances of this class can be picked.
    Future version should maintain unpickling backward-compatability.

    .. code-block:: python

        stopper = HalflifeStopper()
        ...
        while (...):
            stopper.step(score)
            if m.stopper.best_updated:
                # this is the best score we've seen yet
            if not m.stopper.promising:
                # we haven't seen a good score in a long time,
                # and the stopper recommends giving up.
                break

    """
    def __init__(self, 
            initial_wait=20,
            patience_factor=2.0,
            progress_thresh=0.99 ):
        """
        :param method:
        :param method_output_idx:
        :param initial_wait:
        :param patience_factor:
        :param progress_thresh:
        """
        #constants
        self.progress_thresh = progress_thresh
        self.patience_factor = patience_factor
        self.initial_wait = initial_wait

        #dynamic variables
        self.iter = 0
        self.promising = True

        self.halflife_iter = -1
        self.halflife_value = float('inf')
        self.halflife_updated = False

        self.best_iter = -1
        self.best_value = float('inf')
        self.best_updated = False


    def step(self, value):
        if value < (self.halflife_value * self.progress_thresh):
            self.halflife_updated = True
            self.halflife_value = value
            self.halflife_iter = self.iter
        else:
            self.halflife_updated = False

        if value < self.best_value:
            self.best_updated = True
            self.best_value = value
            self.best_iter = self.iter
        else:
            self.best_updated = False

        self.promising = not numpy.isnan(value) \
            and ((self.iter < self.initial_wait) 
                 or (self.iter < (self.halflife_iter * self.patience_factor)))
        self.iter += 1

    def __str__(self):
        return ("Stopper{iter=%(iter)s,"
        "promising=%(promising)s,best_iter=%(best_iter)s,best_value=%(best_value)s"
        ",patience=%(patience_factor)s}")%self.__dict__

class MIREX_Model(object):

    def __init__(self, softmax=False, extra_layer=False):

        self.extract_fns = {}
        self.n_coefs = 32
        self.w_means = None
        self.w_packedcov = None
        self.bias = None
        self.n_frames_per_segment=80
        self.lr = numpy.array(0.001, dtype='float32')
        self.softmax = softmax
        self.extra_layer = extra_layer

    def framelen(self, sr):
        # get some power of 2 that scales linearly-ish with sr
        # it gives 512 for sr=16K 
        return 2**int(numpy.log2(sr / 30.))

    def get_fn_for_sr(self, sr):
        if sr in self.extract_fns:
            return self.extract_fns[sr]

        frames = theano.tensor.dmatrix()
        features = classifaudio.mel_logspec(frames, self.n_coefs, sr, wlen=self.framelen(sr))

        n_rows, n_cols = features.shape
        shape0 = n_rows//self.n_frames_per_segment
        shape1 = self.n_frames_per_segment
        shape2 = n_cols
        segments = theano.tensor.reshape(features[:shape0*shape1], (shape0, shape1, shape2))

        outputs=[
            tensor.cast(classifaudio.segment_mean(segments), 'float32'),
            tensor.cast(classifaudio.segment_packed_cov(segments, width=self.n_coefs),'float32')
            ]

        f = theano.function([frames],outputs)
        self.extract_fns[sr] = f
        return f

    def extract_features(self, samples, sr):

        # downmix to stereo using mean if necessary
        try:
            n_samples, = samples.shape 
        except:
            n_samples, n_channels = samples.shape
            assert n_channels < 5
            assert n_samples > 5
            samples = numpy.mean(samples, axis=1)
        assert n_samples == len(samples)

        framelen = self.framelen(sr)
        samps_per_segment = framelen * self.n_frames_per_segment
        if n_samples < samps_per_segment:
            pad = 1e-5*numpy.random.randn(samps_per_segment-n_samples)
            samples = numpy.hstack((samples, pad))
            n_samples = len(samples)
        n_frames = n_samples // framelen
        frames_shape = (n_frames, framelen)

        frames = (samples[:frames_shape[0]*frames_shape[1]]).reshape(frames_shape)
        fn = self.get_fn_for_sr(int(sr))
        return fn(frames) #returns a pair of matrices

    def normalize_features(self, features):
        # subtract mean off of features:
        # slightly biased by shorter songs probly ok.
        vecs = []
        for f in features:
            for seg_f in f:
                assert seg_f.ndim == 1
                vecs.append(seg_f)
        tmp = numpy.asarray(vecs)
        m = numpy.asarray(tmp.mean(axis=0), dtype='float32')
        s = numpy.asarray(tmp.std(axis=0)+1e-6, dtype='float32')
        tmp -= m
        tmp /= s

        t = 0
        for f in features:
            for i in range(len(f)):
                f[i] = tmp[t]
                t += 1
        return m, s

    def train(self, features, labels):

        self.mean_ms = self.normalize_features([f[0] for f in features])
        self.pack_ms = self.normalize_features([f[1] for f in features])

        n_means = features[0][0].shape[1]
        n_packedcov = features[0][1].shape[1]
        n_labels = labels.shape[1]
        assert len(features) == len(labels)
        n_valid = int(len(features)*.2)
        n_train = len(features) - n_valid

        assert labels.min() >= 0
        assert labels.max() <= 1
        assert n_means == self.n_coefs

        if self.w_means == None:

            dtype = 'float32'

            if self.extra_layer:
                n_hid = 4 * n_labels
            else:
                n_hid = n_labels

            self.w_means = theano.shared(
                    numpy.zeros((n_means, n_hid), dtype=dtype),
                    name='w_means')
            self.w_packedcov = theano.shared(
                    numpy.zeros((n_packedcov, n_hid), dtype=dtype),
                    name='w_packedcov')
            self.bias = theano.shared(
                    numpy.zeros(n_hid, dtype=dtype),
                    name='bias')

            rng = numpy.random.RandomState(234234)

            self.v = theano.shared(
                    numpy.asarray(rng.randn(n_labels, n_labels)*.01,dtype=dtype),
                    name='v')
            self.c = theano.shared(
                    numpy.zeros(n_labels, dtype=dtype),
                    name='bias')

        x_means = theano.tensor.fmatrix()
        x_packedcov = theano.tensor.fmatrix()
        y = theano.tensor.fvector()

        if self.extra_layer:
            frame_predictions = theano.tanh(
                    tensor.add(
                        tensor.dot(x_means, self.w_means),
                        tensor.dot(x_packedcov, self.w_packedcov),
                        self.bias))
            hidfeatures = frame_predictions.mean(axis=0)
            song_prediction = tensor.nnet.softmax(dot(hidfeatures,self.v)+self.c)
            yy = tensor.shape_padleft(y)
            ss = song_prediction.shape_padleft(song_prediction)
            y_idx = tensor.argmax(yy, axis=1)
            cost = tensor.mean(-tensor.log(ss)[tensor.arange(y_idx.shape[0]), y_idx])
            gwm, gwp, gb, gv, gc = tensor.grad(cost, [self.w_means, self.w_packedcov,
                self.bias, self.v, self.c])

            self.predict = theano.function([x_means, x_packedcov], song_prediction)

            valid_fn = theano.function([x_means, x_packedcov, y], cost)
            train_fn = theano.function([x_means, x_packedcov, y], [],
                updates={
                    self.w_means: self.w_means - theano.tensor.cast(self.lr * gwm, dtype=dtype),
                    self.w_packedcov: self.w_packedcov - theano.tensor.cast(self.lr * gwp,dtype=dtype),
                    self.bias: self.bias - theano.tensor.cast(self.lr * gb, dtype=dtype),
                    self.v: self.v - theano.tensor.cast(self.lr * gv, dtype=dtype),
                    self.c: self.c - theano.tensor.cast(self.lr * gc, dtype=dtype),
                    })

        else:

            if self.softmax:
                nonlin = tensor.nnet.softmax
            else:
                nonlin = tensor.nnet.sigmoid

            frame_predictions = nonlin(
                    tensor.add(
                        tensor.dot(x_means, self.w_means),
                        tensor.dot(x_packedcov, self.w_packedcov),
                        self.bias))
            song_prediction = frame_predictions.mean(axis=0)

            eps = 1e-5
            if self.softmax:
                cost = tensor.sum(-y * tensor.log(song_prediction+eps))
            else:
                cost = tensor.sum(\
                    -  y * tensor.log(song_prediction+eps)\
                           - (1-y)*tensor.log(1+eps-song_prediction))
            
            gwm, gwp, gb = tensor.grad(cost, [self.w_means, self.w_packedcov, self.bias])
            

            self.predict = theano.function([x_means, x_packedcov], song_prediction)

            valid_fn = theano.function([x_means, x_packedcov, y], cost)
            train_fn = theano.function([x_means, x_packedcov, y], [],
                updates={
                    self.w_means: self.w_means - theano.tensor.cast(self.lr * gwm, dtype=dtype),
                    self.w_packedcov: self.w_packedcov - theano.tensor.cast(self.lr * gwp,dtype=dtype),
                    self.bias: self.bias - theano.tensor.cast(self.lr * gb, dtype=dtype)})
        
        stopper = HalflifeStopper()
        i = 0
        best = None
        while True:
            j = i % len(features)

            if str(features[j][0].dtype) == 'float64':
                features[j][0] = numpy.asarray(features[j][0], dtype='float32')
            if str(features[j][1].dtype) == 'float64':
                features[j][1] = numpy.asarray(features[j][1], dtype='float32')
            if str(labels[j].dtype) == 'float64':
                labels[j] = numpy.asarray(labels[j], dtype='float32')

            if j == 0:          # start validation
                valid_rvals = []
            if j < n_valid and numpy.prod(features[j][0].shape)> 1:    # start validation
                value = valid_fn(features[j][0],features[j][1], labels[j])
                valid_rvals.append(value)
                #valid_rvals.append(valid_fn(features[j][0], features[j][1], labels[j]))

            if j == n_valid:   # complete validation
                print 'validation cost:',numpy.mean(valid_rvals)
                stopper.step(numpy.mean(valid_rvals))
                if stopper.best_updated:
                    print 'updating best values'
                    best = [
                            self.w_means.value.copy(), 
                            self.w_packedcov.value.copy(), 
                            self.bias.value.copy()]
                if not stopper.promising:
                    break
            if j >= n_valid and numpy.prod(features[j][0].shape) > 1:   # train
                train_fn(features[j][0], features[j][1], labels[j])
            i += 1

        if best:
            # store the best parameters we found back to the shared variables
            print 'saving best values'
            self.w_means.value = best[0]
            self.w_packedcov.value = best[1]
            self.bias.value = best[2]

    def evaluate(self, features, binary=False):
        mm, ms = self.mean_ms #train set mean and std.dev of feature0
        pm, ps = self.pack_ms #train set mean and std.dev of feature1
        rval = numpy.array([self.predict((f[0] - mm)/ ms, (f[1]-pm)/ps)
            for f in features])

        if binary:
            return rval > 0.02
        else:
            return rval



def test_mirex_model():

    model = MIREX_Model()

    import wave
    w = wave.open('./blah.wav')

    print w.getnchannels()
    print w.getsampwidth()
    print w.getnframes() 
    print w.getframerate()

    assert w.getsampwidth() == 2 #bytes -> 16bit
    samples = numpy.frombuffer(w.readframes(w.getnframes()), dtype='int16')
    samples = samples * 1.0 / (2**15)

if __name__=='__main__':
    features = [model.extract_features(samples, w.getframerate()) for i in range(10)]

    model.train(features, numpy.random.rand(10, 7)>.7)

    print model.evaluate(features)

