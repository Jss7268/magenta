# Copyright 2020 Jack Spencer Smith.
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

import tensorflow as tf
from dotmap import DotMap
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from magenta.models.polyamp import constants, sequence_prediction_util
from magenta.models.polyamp.accuracy_util import multi_track_present_accuracy_wrapper, \
    multi_track_prf_wrapper, \
    single_track_present_accuracy_wrapper
from magenta.models.polyamp.layer_util import NoteCroppingsToPianorolls
from magenta.models.polyamp.loss_util import full_model_loss_wrapper
from magenta.models.polyamp.timbre_dataset_util import NoteCropping


def get_default_hparams():
    return {
        'full_learning_rate': 2e-4,
        'multiple_instruments_threshold': 0.6,
        'use_all_instruments': False,
        'melodic_trainable': True,
        'family_recall_weight': [1.0, 1.4, 1.4, .225, .05, 1.2, 1., 1.3, .45, 1.0, 1.0, .7],
    }


class FullModel:
    def __init__(self, melodic_model, timbre_model, hparams):
        if hparams is None:
            hparams = DotMap(get_default_hparams())
        self.hparams = hparams
        self.melodic_model: Model = melodic_model
        self.timbre_model: Model = timbre_model

    def sequence_to_note_croppings(self, sequence):
        """
        Converts a NoteSequence Proto to a list of note_croppings
        :param sequence: NoteSequence to convert
        :return: list of note_croppings generated from sequence
        """
        note_croppings = []
        for note in sequence.notes:
            note_croppings.append(NoteCropping(pitch=note.pitch,
                                               start_idx=note.start_time * self.hparams.sample_rate,
                                               end_idx=note.end_time * self.hparams.sample_rate))
        if len(note_croppings) == 0:
            note_croppings.append(NoteCropping(
                pitch=-1e+7,
                start_idx=-1e+7,
                end_idx=-1e+7
            ))
        return note_croppings

    def get_croppings(self, input_list):
        """
        Convert frame predictions into a sequence. Pad so all batches have same nof notes.
        :param input_list: frames, onsets, offsets
        :return: Tensor of padded cropping lists (padded with large negative numbers)
        """
        batched_frame_predictions, batched_onset_predictions, batched_offset_predictions = \
            input_list

        croppings_list = []
        for batch_idx in range(K.int_shape(batched_frame_predictions)[0]):
            frame_predictions = batched_frame_predictions[batch_idx]
            onset_predictions = batched_onset_predictions[batch_idx]
            offset_predictions = batched_offset_predictions[batch_idx]
            sequence = sequence_prediction_util.predict_sequence(
                frame_predictions=frame_predictions,
                onset_predictions=onset_predictions,
                offset_predictions=offset_predictions,
                velocity_values=None,
                min_pitch=constants.MIN_MIDI_PITCH,
                hparams=self.hparams)
            croppings_list.append(self.sequence_to_note_croppings(sequence))

        padded = tf.keras.preprocessing.sequence.pad_sequences(croppings_list,
                                                               padding='post',
                                                               dtype='int64',
                                                               value=-1e+7)
        return tf.convert_to_tensor(padded)

    def get_full_model(self):
        """Build the Full Model architecture."""
        spec_512 = Input(shape=(None, constants.SPEC_BANDS, 1), name='melodic_spec')
        spec_256 = Input(shape=(None, constants.SPEC_BANDS, 1), name='timbre_spec')
        present_instruments = Input(shape=(self.hparams.timbre_num_classes,))

        # Maybe freeze the layers of the Melodic Model.
        self.melodic_model.trainable = self.hparams.melodic_trainable
        frame_probs, onset_probs, offset_probs = self.melodic_model.call([spec_512])

        stop_gradient_layer = Lambda(lambda x: K.stop_gradient(x))
        frame_predictions = stop_gradient_layer(
            frame_probs > self.hparams.predict_frame_threshold)
        generous_onset_predictions = stop_gradient_layer(
            onset_probs > self.hparams.predict_onset_threshold)
        offset_predictions = stop_gradient_layer(
            offset_probs > self.hparams.predict_offset_threshold)

        note_croppings = Lambda(self.get_croppings,
                                output_shape=(None, 3),
                                dynamic=True,
                                dtype='int64')(
            [frame_predictions, generous_onset_predictions, offset_predictions])

        timbre_probs = self.timbre_model.call([spec_256, note_croppings])

        expand_dims = Lambda(lambda x_list: K.expand_dims(x_list[0], axis=x_list[1]))
        float_cast = Lambda(lambda x: K.cast_to_floatx(x))

        pianoroll = Lambda(lambda x: tf.repeat(K.expand_dims(x),
                                               self.hparams.timbre_num_classes,
                                               -1),
                           output_shape=(None,
                                         constants.MIDI_PITCHES,
                                         self.hparams.timbre_num_classes),
                           dynamic=True)(frame_probs)

        timbre_pianoroll = NoteCroppingsToPianorolls(self.hparams,
                                                     dynamic=True, )(
            [stop_gradient_layer(note_croppings), timbre_probs, stop_gradient_layer(pianoroll)])

        expanded_present_instruments = float_cast(expand_dims([expand_dims([
            (present_instruments), -2]), -2]))

        present_pianoroll = (
            Multiply(name='apply_present')([timbre_pianoroll, expanded_present_instruments]))

        pianoroll_no_gradient = stop_gradient_layer(present_pianoroll)

        # Roll the pianoroll to get instrument predictions for offsets
        # which is normally where we stop the pianoroll fill.
        rolled_pianoroll = Lambda(lambda x: tf.roll(x, 1, axis=-3))(pianoroll_no_gradient)

        expanded_frames = expand_dims([frame_probs, -1])
        expanded_onsets = expand_dims([onset_probs, -1])
        expanded_offsets = expand_dims([offset_probs, -1])

        # Use the last channel for instrument-agnostic midi.
        broadcasted_frames = Concatenate(name='multi_frames')(
            [present_pianoroll, expanded_frames])
        broadcasted_onsets = Concatenate(name='multi_onsets')(
            [present_pianoroll, expanded_onsets])
        broadcasted_offsets = Concatenate(name='multi_offsets')(
            [rolled_pianoroll, expanded_offsets])

        losses = {
            'multi_frames': full_model_loss_wrapper(self.hparams,
                                                    self.hparams.frames_true_weighing),
            'multi_onsets': full_model_loss_wrapper(self.hparams,
                                                    self.hparams.onsets_true_weighing),
            'multi_offsets': full_model_loss_wrapper(self.hparams,
                                                     self.hparams.offsets_true_weighing),
        }

        accuracies = {
            'multi_frames': [
                multi_track_present_accuracy_wrapper(
                    self.hparams.predict_frame_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold),
                single_track_present_accuracy_wrapper(
                    self.hparams.predict_frame_threshold),
                multi_track_prf_wrapper(
                    self.hparams.predict_frame_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold,
                    print_report=True,
                    hparams=self.hparams)
            ],
            'multi_onsets': [
                multi_track_present_accuracy_wrapper(
                    self.hparams.predict_onset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold),
                single_track_present_accuracy_wrapper(
                    self.hparams.predict_onset_threshold),
                multi_track_prf_wrapper(
                    self.hparams.predict_onset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold,
                    print_report=True,
                    hparams=self.hparams)
            ],
            'multi_offsets': [
                multi_track_present_accuracy_wrapper(
                    self.hparams.predict_offset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold),
                single_track_present_accuracy_wrapper(
                    self.hparams.predict_offset_threshold),
                multi_track_prf_wrapper(
                    self.hparams.predict_offset_threshold,
                    multiple_instruments_threshold=self.hparams.multiple_instruments_threshold,
                    hparams=self.hparams)
            ]
        }

        return Model(inputs=[spec_512, spec_256, present_instruments],
                     outputs=[broadcasted_frames, broadcasted_onsets,
                              broadcasted_offsets]), losses, accuracies
