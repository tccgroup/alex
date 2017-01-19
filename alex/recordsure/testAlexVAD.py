#!/usr/bin/env python

import sys
import os
import argparse
from StringIO import StringIO
#import librosa
import numpy as np
#import scipy.io.wavfile as scipyWav
import matplotlib.pyplot as plt

pathToAdd = os.path.join(os.path.dirname(sys.argv[0]), "../../../experiments/share/")
sys.path.append(pathToAdd)

from lsdpylib import phraselib
from lsdpylib import lsdutils


if __name__ == '__main__':
    import autopath

from alex.utils import htk
from alex.tools.vad import train_vad_nn_theano as train
from alex.ml import ffnn
from alex.ml import tffnn


def mediumTermeWindow(prev_frames=15, next_frames=15, amplify_center_frame=4.0):

    amp_min = 1.0 / float(amplify_center_frame)
    amp_max = 1.0
    amp_prev = [amp_min + (float(i) / prev_frames) * (amp_max - amp_min) for i in range(0, prev_frames)]
    amp_next = [amp_min + (float(i) / next_frames) * (amp_max - amp_min) for i in range(next_frames-1, -1, -1 )]

    return np.repeat(amp_prev  + [1.0] + amp_next, 1000 / (prev_frames + 1 + next_frames))


def tmp_mlf(fInAudioName, pOut):

    fName = lsdutils.getFileBasename(fInAudioName)
    fOutMLFName = os.path.join(pOut, fName + ".tmp.mlf")

    length = lsdutils.audioLength(fInAudioName)
    phrase = phraselib.Phrase(start=0, end=length)
    phrase.setWords("unknown")

    phraselib.writeMLF( [(fInAudioName, [phrase])], filename=fOutMLFName)

    return fOutMLFName


def extractFeatures(fInAudioName, keep_mlf=False):

    next_frames = 15
    prev_frames = 15
    amplify_center_frame = 4.0
    usec0=0
    usedelta=False
    useacc=False
    mel_banks_only=1

    fOutMLFName = tmp_mlf(fInAudioName, pOut)
    length = lsdutils.audioLength(fInAudioName)
    mlf = train.load_mlf(fOutMLFName, 1, 1+int(length*100))

    vta = htk.MLFMFCCOnlineAlignedArray(usec0=usec0,n_last_frames=0, usedelta = usedelta, useacc = useacc, mel_banks_only = mel_banks_only)
    vta.append_mlf(mlf)

    pIn = os.path.dirname(fInAudioName)
    vta.append_trn(os.path.join(pIn, '*.wav'))

    mfcc = vta.__iter__().next()

    test_x = np.array([frame for frame, _ in vta]).astype(np.float32)

    if keep_mlf is False:
        os.remove(fOutMLFName)

    return test_x


def saveFeatures(fOutFeatureName, features):

#    fOutFeatureName = os.path.join(pOut, "mao.npy")
    print 'Saving data to:', fOutFeatureName
#    features, mlf = feats
#    fName = lsdutils.getFileBasename(fInAudioName)
    with open(fOutFeatureName, 'wb') as fOutFeature:
        np.save(fOutFeature, features, allow_pickle=False)
#        np.save(fOutFeature, mlf, allow_pickle=False)


def test_model(features, fInModelName, pOut, ID=None):

    fOutVADName = os.path.join(pOut, fName + ".vad")

    e = tffnn.TheanoFFNN()
    e.load(fInModelName)

    features_m = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    features -= features_m
    features /= features_std

    e.set_input_norm(features_m, features_std)

    predictions_y = e.predict(features, batch_size=100000, prev_frames= 15, next_frames=15)

    with open(fOutVADName, 'w') as fOutVAD:
        for p in predictions_y:
            print >> fOutVAD, "%.3f,%.3f" % (p[0], p[1])

    return predictions_y


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracts features using ALEX.')

    parser.add_argument('-i','--input', help='Either audio or dnn features input file.', required=True)
    parser.add_argument('-m','--model', help='Input model file.', default=None)
#    parser.add_argument('-l','--list', help='Input list of audio files.', default=None)
    parser.add_argument('-o','--outdir', help='Output directory', required=True)
    parser.add_argument('--plot', help='Plot results', default=False, action='store_true')
    parser.add_argument('--keep-mlf', help='Keep temporary mlf file', default=False, action='store_true')
    
#    parser.add_argument('--logdir', help='Log file(s) directory path. Log file will contain all the parsing warnings and some transcript stats.', nargs='?', const=True, default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    fInInputName = args.input
    fInModelName = args.model
#    fInListName = args.list
    pOut = args.outdir
    doPlot = args.plot
    keep_mlf = args.keep_mlf

#    window = mediumTermeWindow()

    fName = lsdutils.getFileBasename(fInInputName)

    if fInInputName.endswith('.wav') :
        features = extractFeatures(fInInputName, keep_mlf=keep_mlf)

        fOutMLFName = os.path.join(pOut, fName + ".dnnfeats")
        saveFeatures(fOutMLFName, features)

    elif fInInputName.endswith('.dnnfeats'):
        with open(fInInputName, 'rb') as fInInput:
            features = np.load(fInInput)

    print features.shape

    if fInModelName is not None:

        predictions_y = test_model(features, fInModelName, pOut, ID=fName)

        if doPlot is not False:
            plt.plot(predictions_y[:,0], 'r')
            plt.plot(predictions_y[:,1], 'b')
            plt.show()

