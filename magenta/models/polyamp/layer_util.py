# Copyright 2020 A. Anonymous. - The PolyAMP Authors.
#
# We have removed the license from our anonymized code.
# All rights reserved.

"""A utility for different layers in the Neural Networks."""
import functools
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K, layers
from tensorflow.keras.layers import Conv2D, ELU, MaxPooling2D

from magenta.models.polyamp import constants, sequence_prediction_util
from magenta.models.polyamp.timbre_dataset_util import NoteCropping, get_cqt_index, get_mel_index
from magenta.music import midi_io


def elu_pool_layer(pool_size, activation_fn=ELU()):
    """ELU activation, then max-pool."""

    def elu_pool_fn(x):
        return MaxPooling2D(pool_size=pool_size)(activation_fn(x))

    return elu_pool_fn


def conv_elu_pool_layer(num_filters, conv_temporal_size, conv_freq_size, pool_size,
                        activation_fn=ELU()):
    """Conv2D, ELU activation, then max-pool."""

    def conv_elu_pool_fn(x):
        return elu_pool_layer(pool_size, activation_fn=activation_fn)(
            Conv2D(num_filters,
                   [conv_temporal_size, conv_freq_size],
                   padding='same',
                   use_bias=False,
                   kernel_initializer='he_uniform')(x))

    return conv_elu_pool_fn


def normalize_and_weigh(inputs, num_notes, pitches, hparams):
    """Decrease the values higher above the fundamental frequency."""
    gradient_pitch_mask = 1 + K.int_shape(inputs)[-2] - K.arange(K.int_shape(inputs)[-2])
    gradient_pitch_mask = gradient_pitch_mask / K.max(gradient_pitch_mask)
    gradient_pitch_mask = K.expand_dims(K.cast_to_floatx(gradient_pitch_mask), 0)
    gradient_pitch_mask = tf.repeat(gradient_pitch_mask, axis=0, repeats=num_notes)
    gradient_pitch_mask = (gradient_pitch_mask
                           + K.expand_dims(pitches / K.int_shape(inputs)[-2], -1))
    exp = (math.log(hparams.timbre_gradient_exp) if hparams.timbre_spec_log_amplitude
           else hparams.timbre_gradient_exp)
    gradient_pitch_mask = tf.minimum(gradient_pitch_mask ** exp, 1.0)
    gradient_pitch_mask = K.expand_dims(gradient_pitch_mask, -1)
    gradient_product = inputs * gradient_pitch_mask
    return gradient_product


def get_all_croppings(input_list, hparams):
    """This increases dimensionality by duplicating our input spec
    image and getting differently cropped views of it.
    Essentially, get a small view for each note being played in a piece.
    """
    conv_output_list = input_list[0]
    note_croppings_list = input_list[1]

    all_outputs = []
    # Un-batch and do different things for each batch.
    # This creates mini-mini-batches).
    for batch_idx in range(K.int_shape(conv_output_list)[0]):
        if K.int_shape(note_croppings_list)[1] == 0:
            out = np.zeros(shape=(1, K.int_shape(conv_output_list[batch_idx])[1],
                                  K.int_shape(conv_output_list[batch_idx])[-1]))
        else:
            out = get_croppings_for_single_image(
                conv_output_list[batch_idx],
                note_croppings_list[batch_idx],
                hparams=hparams,
                temporal_scale=max(1, (hparams.timbre_pool_size[0][0]
                                       * hparams.timbre_pool_size[1][0])))

        all_outputs.append(out)

    return tf.convert_to_tensor(all_outputs)


def get_croppings_for_single_image(conv_output, note_croppings,
                                   hparams=None, temporal_scale=1.0):
    """Separate the note regions for an individual spectrogram.
    A high-pass filter removes the effect of values below
    the fundamental frequency.
    """
    num_notes = K.int_shape(note_croppings)[0]
    pitch_idx_fn = functools.partial(
        get_cqt_index if hparams.timbre_spec_type == 'cqt' else get_mel_index,
        hparams=hparams)
    pitch_to_spec_index = tf.map_fn(
        pitch_idx_fn,
        tf.gather(note_croppings, indices=0, axis=1))
    gathered_pitches = (K.cast_to_floatx(pitch_to_spec_index)
                        * K.int_shape(conv_output)[1]
                        / constants.TIMBRE_SPEC_BANDS)
    pitch_mask = K.expand_dims(
        K.cast(tf.where(tf.sequence_mask(
            K.cast(
                gathered_pitches, dtype='int32'
            ), K.int_shape(conv_output)[1]
            # Don't lose gradient completely so multiply by 2e-3.
        ), 2e-3, 1), tf.float32), -1)

    trimmed_list = []
    start_idx = K.cast(
        tf.gather(note_croppings, indices=1, axis=1)
        / hparams.timbre_hop_length
        / temporal_scale, dtype='int32'
    )
    end_idx = K.cast(
        tf.gather(note_croppings, indices=2, axis=1)
        / hparams.timbre_hop_length
        / temporal_scale, dtype='int32'
    )
    for i in range(num_notes):
        if end_idx[i] < 0:
            # This must be a padded value note.
            trimmed_list.append(
                np.zeros(shape=(1, K.int_shape(conv_output)[1], K.int_shape(conv_output)[2]),
                         dtype=K.floatx()))
        else:
            trimmed_spec = conv_output[min(start_idx[i], K.int_shape(conv_output)[0] - 1)
                                       :max(end_idx[i], start_idx[i] + 1)]
            max_pool = K.max(trimmed_spec, 0)
            trimmed_list.append(K.expand_dims(max_pool, 0))

    broadcasted_spec = K.concatenate(trimmed_list, axis=0)

    mask = broadcasted_spec * pitch_mask
    return normalize_and_weigh(mask, num_notes, gathered_pitches, hparams)


class NoteCroppingsToPianorolls(layers.Layer):
    def __init__(self, hparams, **kwargs):
        self.hparams = hparams
        super(NoteCroppingsToPianorolls, self).__init__(**kwargs)

    def call(self, input_list, **kwargs):
        """
        Convert note croppings and their corresponding timbre
        predictions to a pianoroll that
        we can multiply by the melodic predictions.
        :param input_list: note_croppings, timbre_probs, pianorolls
        :return: a pianoroll with shape:
        (batches, pianoroll_length, 88, timbre_num_classes + 1)
        """
        batched_note_croppings, batched_timbre_probs, batched_pianorolls = input_list

        pianoroll_list = []
        for batch_idx in range(K.int_shape(batched_note_croppings)[0]):
            note_croppings = batched_note_croppings[batch_idx]
            timbre_probs = batched_timbre_probs[batch_idx]

            pianorolls = K.zeros(
                shape=(K.int_shape(batched_pianorolls[batch_idx])[0],
                       constants.MIDI_PITCHES,
                       self.hparams.timbre_num_classes))
            ones = np.ones(
                shape=(K.int_shape(batched_pianorolls[batch_idx])[0],
                       constants.MIDI_PITCHES,
                       self.hparams.timbre_num_classes))

            for i, note_cropping in enumerate(note_croppings):
                cropping = NoteCropping(*note_cropping)
                pitch = cropping.pitch - constants.MIN_MIDI_PITCH
                if cropping.end_idx < 0:
                    # Don't fill padded notes.
                    continue
                start_idx = K.cast(cropping.start_idx / self.hparams.spec_hop_length, 'int64')
                end_idx = K.cast(cropping.end_idx / self.hparams.spec_hop_length, 'int64')
                pitch_mask = K.cast_to_floatx(tf.one_hot(pitch, constants.MIDI_PITCHES))
                end_time_mask = K.cast(tf.sequence_mask(
                    end_idx,
                    maxlen=K.int_shape(batched_pianorolls[batch_idx])[0]
                ), tf.float32)
                start_time_mask = K.cast(tf.math.logical_not(tf.sequence_mask(
                    start_idx,
                    maxlen=K.int_shape(batched_pianorolls[batch_idx])[0]
                )), tf.float32)
                time_mask = start_time_mask * end_time_mask
                # Constant time for the pitch mask.
                pitch_mask = K.expand_dims(K.expand_dims(pitch_mask, 0))
                # Constant pitch for the time mask.
                time_mask = K.expand_dims(K.expand_dims(time_mask, 1))
                mask = ones * pitch_mask
                mask = mask * time_mask
                cropped_probs = mask * (timbre_probs[i])
                if K.learning_phase() == 1:
                    # For training, this is necessary for the gradient.
                    pianorolls = pianorolls + cropped_probs
                else:
                    # For testing, this is faster.
                    pianorolls.assign_add(cropped_probs)

            frame_predictions = pianorolls > self.hparams.multiple_instruments_threshold
            sequence = sequence_prediction_util.predict_multi_sequence(
                frame_predictions=frame_predictions,
                min_pitch=constants.MIN_MIDI_PITCH,
                hparams=self.hparams)
            midi_filename = (
                f'./out/{batch_idx}-of-{K.int_shape(batched_note_croppings)[0]}.midi'
            )
            midi_io.sequence_proto_to_midi_file(sequence, midi_filename)
            # Make time the first dimension.
            pianoroll_list.append(pianorolls)

        return tf.convert_to_tensor(pianoroll_list)

    def compute_output_shape(self, input_shape):
        # Output shape will be the same as the input pianoroll.
        return input_shape[2]
