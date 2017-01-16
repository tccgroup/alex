#!/usr/bin/env python

import argparse
import sys
import os
import shutil
import wave
import logging
import datetime
import random
import numpy as np

if __name__ == '__main__':
    import autopath


from alex.utils.htk import *
from alex.ml import ffnn
from alex.ml import tffnn

from alex.components.vad.ffnn import FFNNVADGeneral
from alex.ml.tffnn import TheanoFFNN

weight_l2=1e-6
batch_size= 500000 

max_frames = 1000000 #1000000

max_files = 1000000
max_frames_per_segment = 0
trim_segments = 0

next_frames = 15
prev_frames = 15

usec0=0
usedelta=False
useacc=False
mel_banks_only=1
uselda = 0


global features_file_name


def load_mlf(train_data_sil_aligned, max_files, max_frames_per_segment):
    """ Loads a MLF file and creates normalised MLF class.

    :param train_data_sil_aligned:
    :param max_files:
    :param max_frames_per_segment:
    :return:
    """
    print "Loading %s" % train_data_sil_aligned
    mlf = MLF(train_data_sil_aligned, max_files=max_files)
    mlf.filter_zero_segments()
    # map all sp, _noise_, _laugh_, _inhale_ to sil
    mlf.sub('sp', 'sil')
    mlf.sub('_noise_', 'sil')
    mlf.sub('_laugh_', 'sil')
    mlf.sub('_inhale_', 'sil')
    mlf.sub('SIL', 'sil')
    mlf.sub('SP', 'sil')
    mlf.sub('_NOISE_', 'sil')
    mlf.sub('_LAUGH_', 'sil')
    mlf.sub('_INHALE_', 'sil')
    # map everything except of sil to speech
    mlf.sub('sil', 'speech', False)
    mlf.merge()
    #mlf_sil.times_to_seconds()
    mlf.times_to_frames()
    mlf.trim_segments(trim_segments)
    mlf.shorten_segments(max_frames_per_segment)

    return mlf


def gen_features(speech_data, speech_alignment):
    vta = MLFMFCCOnlineAlignedArray(usec0=usec0,n_last_frames=0, usedelta = usedelta, useacc = useacc, mel_banks_only = mel_banks_only)
    sil_count = 0
    speech_count = 0
    for sd, sa in zip(speech_data, speech_alignment):
        mlf_speech = load_mlf(sa, max_files, max_frames_per_segment)
        vta.append_mlf(mlf_speech)
        vta.append_trn(sd)

        sil_count += mlf_speech.count_length('sil')
        speech_count += mlf_speech.count_length('speech')

    print "The length of sil segments:    ", sil_count
    print "The length of speech segments: ", speech_count

    mfcc = vta.__iter__().next()

    print "Features vector length:", len(mfcc[0])
    input_size = len(mfcc[0])

    crossvalid_x = []
    crossvalid_y = []
    train_x = []
    train_y = []
    i = 0
    for frame, label in vta:
        # downcast
        if frame is not None:
            frame = frame.astype(np.float32)
        #        frame = frame - (10.0 if mel_banks_only else 0.0)

            crossvalid_x.append(frame)
            if label == "sil":
                crossvalid_y.append(0)
            else:
                crossvalid_y.append(1)

    crossvalid_x = np.array(crossvalid_x).astype(np.float32)
    crossvalid_y = np.array(crossvalid_y).astype('int32')
#    train_x = np.array(train_x).astype(np.float32)
#    train_y = np.array(train_y).astype('int32')


    # normalise the data
    tx_m = np.mean(crossvalid_x, axis=0)
    tx_std = np.std(crossvalid_x, axis=0)

    crossvalid_x -= tx_m
    crossvalid_x /= tx_std


    print 'Saving data to:', features_file_name
    f = open(features_file_name, "wb")
    np.save(f, crossvalid_x) 
    np.save(f, crossvalid_y)
    np.save(f, train_x)
    np.save(f, train_y)
    np.save(f, tx_m)
    np.save(f, tx_std)
    f.close()

    return crossvalid_x, crossvalid_y, train_x, train_y, tx_m, tx_std


def get_accuracy(true_y, predictions_y):
    """ Compute accuracy of predictions from the activation of the last NN layer, and the sil prior probability.

    :param ds: the training dataset
    :param a: activation from the NN using the ds datasat
    """
    acc = np.mean(np.equal(np.argmax(predictions_y, axis=1),true_y))*100.0
    sil = (1.0-float(np.count_nonzero(true_y)) / len(true_y))*100.0

    return acc, sil


def test(fInModelName, speech_data, speech_alignment):
    print
    print datetime.datetime.now()
    print
    random.seed(0)


    try:
        f = open(features_file_name, "rb")
        crossvalid_x = np.load(f) 
        crossvalid_y = np.load(f)
        train_x = np.load(f)
        train_y = np.load(f)
        tx_m = np.load(f)
        tx_std = np.load(f)
        f.close()
    except IOError:
        print speech_data
        print speech_alignment
        print features_file_name
        crossvalid_x, crossvalid_y, train_x, train_y, tx_m, tx_std = gen_features(speech_data, speech_alignment)

    input_size = crossvalid_x.shape[1] * (prev_frames + 1 + next_frames)

    tx_m = np.tile(tx_m, prev_frames + 1 + next_frames)
    tx_std = np.tile(tx_std, prev_frames + 1 + next_frames)

    e = tffnn.TheanoFFNN()
    e.load(fInModelName)
    e.set_input_norm(tx_m, tx_std)

    predictions_y, gold_y = e.predict(crossvalid_x, batch_size, prev_frames, next_frames, crossvalid_y)


    return predictions_y, gold_y

def saveVAD(fOutVADName, predictions_y, gold_y=None):

    if gold_y is None:
        gold_y =['unk'] * len(predictions_y)

    with open(fOutVADName, 'w') as fOutVAD:
        for p, g in zip(predictions_y, gold_y):
            print >> fOutVAD, "{:.3f},{:.3f},{}".format(p[0], p[1], g)


if __name__ == '__main__':


    global crossvalid_frames, usec0

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help="Input audio file", required=True)
    parser.add_argument('--mlf', help="Input mlf file", default=None)
    parser.add_argument('-m', '--model', help="Input model file", required=True)
    parser.add_argument('-o', '--outdir', help="Output directory", required=True)

    args = parser.parse_args()

    pOut = args.outdir
    fInAudioName = args.input
    fInMLFName = args.mlf
    fInModelName = args.model

#    crossvalid_frames = int((0.20 * max_frames))

    features_file_name = "%s/vad_sds_mfcc_mfr%d_mfl%d_mfps%d_ts%d_usec0%d_usedelta%d_useacc%d_mbo%d.npc" % \
                         (pOut, max_frames, max_files, max_frames_per_segment, trim_segments, usec0, usedelta, useacc, mel_banks_only)


    fOutVADName = os.path.join(pOut, "results.vad")

    train_speech = [fInAudioName]
    train_speech_alignment = [fInMLFName]

    predictions_y, gold_y = test(fInModelName, train_speech, train_speech_alignment)

    c_acc, c_sil = get_accuracy(gold_y, predictions_y)

    saveVAD(fOutVADName, predictions_y, gold_y)


