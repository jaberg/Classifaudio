"""Python module defining Ops and functions for using Theano to classify audio.
"""
import mel, audiofeature, segmentfeatures

from audiofeature import (
        AudioFeatures,
        mfcc_htk,
        mel_logspec
        )

from segmentfeatures import (
        segment_mean, 
        segment_var,
        segment_packed_cov,
        segment_packed_corr)
