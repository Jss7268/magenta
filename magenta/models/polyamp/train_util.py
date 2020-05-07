# Copyright 2020 The Magenta Authors.
# Modifications Copyright 2020 Jack Smith.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for training."""

import copy
import functools
import glob
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from magenta.models.polyamp import constants
from magenta.models.polyamp.callback import EvaluationMetrics, \
    MidiPredictionMetrics
from magenta.models.polyamp.data_generator import DataGenerator
from magenta.models.polyamp.dataset_reader import wav_to_spec_op
from magenta.models.polyamp.model_util import ModelType, ModelWrapper
from magenta.models.polyamp.timbre_dataset_reader import create_spectrogram
from magenta.music import audio_io, midi_io


def train(data_fn,
          model_dir,
          model_type,
          preprocess_examples,
          hparams,
          num_steps=50):
    """Train loop."""

    transcription_data = functools.partial(
        data_fn,
        preprocess_examples=preprocess_examples,
        is_training=True,
        shuffle_examples=True,
        skip_n_initial_records=random.randint(0, 128))

    model_wrapper = ModelWrapper(model_dir, model_type, id_=hparams.model_id,
                                 dataset=transcription_data(params=hparams),
                                 batch_size=hparams.batch_size,
                                 steps_per_epoch=hparams.epochs_per_save,
                                 hparams=hparams)

    if model_type == ModelType.MELODIC:
        model_wrapper.build_model()
        model_wrapper.load_newest(hparams.load_id)
    elif model_type == ModelType.TIMBRE:
        model_wrapper.build_model()
        model_wrapper.load_newest(hparams.load_id)
    else:
        print('building full model')
        melodic_model_wrapper = ModelWrapper(model_dir, ModelType.MELODIC, hparams=hparams)
        melodic_model_wrapper.build_model(compile=False)
        melodic_model_wrapper.load_newest()
        timbre_model_wrapper = ModelWrapper(model_dir, ModelType.TIMBRE, hparams=hparams)
        timbre_model_wrapper.build_model(compile=False)
        timbre_model_wrapper.load_newest()

        model_wrapper.build_model(midi_model=melodic_model_wrapper.get_model(),
                                  timbre_model=timbre_model_wrapper.get_model())

        model_wrapper.load_newest(hparams.load_id)

    for i in range(num_steps):
        model_wrapper.train_and_save(epoch_num=i)


def transcribe(data_fn,
               model_dir,
               model_type,
               path,
               file_suffix,
               hparams):
    if data_fn:
        transcription_data = data_fn(preprocess_examples=True,
                                     is_training=False,
                                     shuffle_examples=True,
                                     skip_n_initial_records=0,
                                     params=hparams)
    else:
        transcription_data = None

    if model_type == ModelType.MELODIC:
        melodic_model_wrapper = ModelWrapper(model_dir, ModelType.MELODIC,
                                             dataset=transcription_data,
                                             batch_size=1, id_=hparams.model_id, hparams=hparams)
        melodic_model_wrapper.build_model(compile=False)
        melodic_model_wrapper.load_newest(hparams.load_id)
    elif model_type == ModelType.TIMBRE:
        timbre_model_wrapper = ModelWrapper(model_dir, ModelType.TIMBRE, id_=hparams.model_id,
                                            dataset=transcription_data, batch_size=1,
                                            hparams=hparams)
        timbre_model_wrapper.build_model(compile=False)
        timbre_model_wrapper.load_newest(hparams.load_id)

    if data_fn:
        while True:
            # This will exit when the dataset runs out.
            # Generally, just predict on filenames rather than
            # TFRecords so you don't use this code.
            if model_type is ModelType.MELODIC:
                x, _ = melodic_model_wrapper.generator.get()
                sequence_prediction = melodic_model_wrapper.predict_from_spec(x[0])
                midi_filename = path + file_suffix + '.midi'
                midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
            elif model_type is ModelType.TIMBRE:
                x, y = timbre_model_wrapper.generator.get()
                timbre_prediction = K.get_value(timbre_model_wrapper.predict_from_spec(*x))[0]
                print(f'True: {x[1][0][0]}{constants.FAMILY_IDX_STRINGS[np.argmax(y[0][0])]}. '
                      f'Predicted: {constants.FAMILY_IDX_STRINGS[timbre_prediction]}')
    else:
        filenames = glob.glob(path)

        for filename in filenames:
            wav_data = tf.io.gfile.GFile(filename, 'rb').read()

            if model_type == ModelType.MELODIC:
                spec = wav_to_spec_op(wav_data, hparams=hparams)

                # Add "batch" and channel dims.
                spec = tf.reshape(spec, (1, *spec.shape, 1))
                sequence_prediction = melodic_model_wrapper.predict_from_spec(spec)
                midi_filename = filename + file_suffix + '.midi'
                midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
            elif model_type == ModelType.TIMBRE:
                y = audio_io.wav_data_to_samples(wav_data, hparams.sample_rate)
                spec = create_spectrogram(K.constant(y), hparams)
                # Add "batch" and channel dims.
                spec = K.cast_to_floatx(tf.reshape(spec, (1, *spec.shape, 1)))
                timbre_prediction = K.get_value(timbre_model_wrapper.predict_from_spec(spec))[0]
                print(f'File: {filename}. '
                      f'Predicted: {constants.FAMILY_IDX_STRINGS[timbre_prediction]}')


def evaluate(data_fn, model_dir, model_type, preprocess_examples, hparams, num_steps=None,
             note_based=False):
    """Evaluation loop."""
    hparams.batch_size = 1
    hparams.slakh_batch_size = 1

    transcription_data_base = functools.partial(
        data_fn,
        preprocess_examples=preprocess_examples,
        is_training=False)

    transcription_data = functools.partial(
        transcription_data_base,
        shuffle_examples=False, skip_n_initial_records=0)

    model_wrapper = ModelWrapper(model_dir, ModelType.FULL,
                                 dataset=transcription_data(params=hparams),
                                 batch_size=hparams.batch_size,
                                 steps_per_epoch=hparams.epochs_per_save,
                                 hparams=hparams)
    if model_type is ModelType.TIMBRE:
        timbre_model = ModelWrapper(model_dir, ModelType.TIMBRE, hparams=hparams)
        timbre_model.build_model(compile=False)
        model_wrapper.build_model(compile=False, timbre_model=timbre_model.get_model())
        model_wrapper.load_newest(hparams.load_id)
        model_wrapper = timbre_model
    elif model_type is ModelType.MELODIC:
        midi_model = ModelWrapper(model_dir, ModelType.MELODIC, hparams=hparams, batch_size=1)
        midi_model.build_model(compile=False)
        model_wrapper.build_model(compile=False, midi_model=midi_model.get_model())
        midi_model.load_newest(hparams.load_id)
        model_wrapper = midi_model
    else:
        model_wrapper.build_model(compile=False)
        model_wrapper.load_newest(hparams.load_id)

    generator = DataGenerator(transcription_data(params=hparams), hparams.batch_size,
                              use_numpy=False)
    save_dir = f'{model_dir}/{model_type.name}/{model_wrapper.id}_eval'

    if model_type is ModelType.MELODIC:
        metrics = MidiPredictionMetrics(generator=generator, note_based=note_based,
                                        hparams=hparams, save_dir=save_dir)
    else:
        metrics = EvaluationMetrics(generator=generator, hparams=hparams, save_dir=save_dir,
                                    is_full=model_type is ModelType.FULL)
    try:
        for i in range(num_steps):
            print(f'evaluating step: {i}')
            metrics.on_epoch_end(i, model=model_wrapper.get_model())
    except:
        pass

    metric_names = ['true_positives', 'false_positives', 'false_negatives']

    if model_type is not ModelType.MELODIC:
        # These are the instrument-specific metrics.
        instrument_true_positives, instrument_false_positives, instrument_false_negatives = [
            functools.reduce(
                lambda a, b: a + b,
                map(lambda x: np.array(
                    [x[constants.FAMILY_IDX_STRINGS[i]][n] for i in range(len(x.keys()))]),
                    metrics.metrics_history)) for n in metric_names
        ]

        instrument_precision = (instrument_true_positives
                                / (instrument_true_positives + instrument_false_positives + 1e-9))
        instrument_recall = (instrument_true_positives
                             / (instrument_true_positives + instrument_false_negatives + 1e-9))

        instrument_f1_score = 2 * ((instrument_precision * instrument_recall)
                                   / (instrument_precision + instrument_recall + 1e-9))

        overall_precision = (np.sum(instrument_true_positives[:-1])
                             / np.sum(instrument_true_positives[:-1]
                                      + instrument_false_positives[:-1] + 1e-9))
        overall_recall = (np.sum(instrument_true_positives[:-1])
                          / np.sum(instrument_true_positives[:-1]
                                   + instrument_false_negatives[:-1] + 1e-9))

        overall_f1_score = 2 * ((overall_precision * overall_recall)
                                / (overall_precision + overall_recall + 1e-9))

        for i in range(hparams.timbre_num_classes + (1 if model_type is ModelType.FULL else 0)):
            instrument = constants.FAMILY_IDX_STRINGS[i]
            print(f'{instrument}: '
                  f'P: {instrument_precision[i]}, '
                  f'R: {instrument_recall[i]}, '
                  f'F1: {instrument_f1_score[i]}, '
                  f'N: {instrument_true_positives[i] + instrument_false_negatives[i]}')
        total_support = K.sum(instrument_true_positives) + K.sum(instrument_false_negatives)
        print(f'overall: '
              f'P: {overall_precision}, '
              f'R: {overall_recall}, '
              f'F1: {overall_f1_score}, '
              f'N: {total_support}')
    elif note_based:
        macro_names = ['note_precision', 'note_recall', 'note_f1_score', 'frame_precision',
                       'frame_recall', 'frame_f1_score']
        note_precision, note_recall, note_f1, frame_precision, frame_recall, frame_f1 = [
            functools.reduce(
                lambda a, b: a + b,
                map(lambda x: [x[n]],
                    metrics.metrics_history)) for n in macro_names
        ]
        print(f'nP: {np.mean(note_precision)}, '
              f'nR: {np.mean(note_recall)}, '
              f'nF: {np.mean(note_f1)}, '
              f'fP: {np.mean(frame_precision)}, '
              f'fR: {np.mean(frame_recall)}, '
              f'fF: {np.mean(frame_f1)}, ')
    else:
        stacks = ['frames', 'onsets', 'offsets']
        # instrument-agnostic metrics

        true_positives, false_positives, false_negatives = [
            functools.reduce(
                lambda a, b: a + b,
                map(lambda x: np.array(
                    [x[stacks[i]][n] for i in range(len(x.keys()))]),
                    metrics.metrics_history)) for n in metric_names
        ]
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)

        f1_score = 2 * precision * recall / (precision + recall + 1e-9)
        support = true_positives + false_negatives
        for i in range(len(stacks)):
            stack = stacks[i]
            print(f'{stack}: '
                  f'P: {precision[i]}, '
                  f'R: {recall[i]}, '
                  f'F1: {f1_score[i]}, '
                  f'N: {support[i]}')
