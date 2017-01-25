#!/usr/bin/env python

import os
import matplotlib

if 'DISPLAY' in os.environ:
    force_save = False
else:
    matplotlib.use('Agg')
    force_save = True


import sys
import argparse
from StringIO import StringIO
#import librosa
import numpy as np
#import scipy.io.wavfile as scipyWav
import matplotlib.pyplot as plt

pathToAdd = os.path.join(os.path.dirname(sys.argv[0]), "../../../experiments/share/")
pathToAdd = os.path.join(os.path.dirname(sys.argv[0]), "../../../experiments-master-live/share/")
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


def extractFeatures(fInInputName, keep_mlf=False, max_frames=10000000):

    next_frames = 15
    prev_frames = 15
    amplify_center_frame = 4.0
    usec0=0
    usedelta=False
    useacc=False
    mel_banks_only=1

    vta = htk.MLFMFCCOnlineAlignedArray(usec0=usec0,n_last_frames=0, usedelta = usedelta, useacc = useacc, mel_banks_only = mel_banks_only)

    if fInInputName.endswith('.wav'):
        fOutMLFName = tmp_mlf(fInInputName, pOut)
        length = lsdutils.audioLength(fInInputName)
        mlf = train.load_mlf(fOutMLFName, 1, 1+int(length*100))

        pIn = os.path.dirname(fInAudioName)
        vta.append_trn(os.path.join(pIn, '*.wav'))

    elif fInInputName.endswith('.mlf'):
        fOutMLFName = None
        mlf = train.load_mlf(fInInputName, 1000000, max_frames)

    vta.append_mlf(mlf)

#    mfcc = vta.__iter__().next()

    if keep_mlf is False and os.path.isfile(fOutMLFName):
        os.remove(fOutMLFName)

    test_x = np.array([frame for frame, _ in vta]).astype(np.float32)
    label_x = np.array([label for _, label in vta])

    return test_x, label_x


def saveFeatures(fOutFeatureName, (features, labels) ):

    print 'Saving data to:', fOutFeatureName

    with open(fOutFeatureName, 'wb') as fOutFeature:
        np.save(fOutFeature, features, allow_pickle=False)
        np.save(fOutFeature, labels, allow_pickle=False)


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
    parser.add_argument('--save', help='Save figure instead of plot to screen.', default=False, action='store_true')
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
    plot_fig = args.plot
    save_fig = args.save
    keep_mlf = args.keep_mlf

#    window = mediumTermeWindow()

    fName = lsdutils.getFileBasename(fInInputName)

    if fInInputName.endswith('.wav') or fInInputName.endswith('.mlf'):
        features, labels = extractFeatures(fInInputName, keep_mlf=keep_mlf)

        fOutMLFName = os.path.join(pOut, fName + ".dnnfeats")
        saveFeatures(fOutMLFName, (features, labels))

    elif fInInputName.endswith('.dnnfeats'):
        with open(fInInputName, 'rb') as fInInput:
            features = np.load(fInInput)
            labels = np.load(fInInput)

    print features.shape

    if fInModelName is not None:

        predictions_y = test_model(features, fInModelName, pOut, ID=fName)
        acc, sil = train.get_accuracy(labels, predictions_y)

        print "ACC: {}, SIL: {}".format(acc, sil)

        if plot_fig is not False:
            fig = plt.figure()

            plt.plot(predictions_y[:,0], 'r')
            plt.plot(predictions_y[:,1], 'b')

            if force_save is True or save_fig is True:
                fPNGMLFName = os.path.join(pOut, fName + ".png")
                fig.savefig(fPNGMLFName)

            else:
                plt.show()

