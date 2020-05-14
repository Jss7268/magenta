# Copyright 2020 The Magenta Authors.
# Modifications Copyright 2020 A. Anonymous. - The PolyAMP Authors.
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

import collections
import functools

import librosa
import numpy as np
import tensorflow.compat.v2 as tf

from magenta.models.polyamp import constants, instrument_family_mappings
from magenta.music import audio_io
from magenta.music.protobuf import music_pb2

FeatureTensors = collections.namedtuple('FeatureTensors', ('spec', 'note_croppings'))

LabelTensors = collections.namedtuple('LabelTensors', ('instrument_families',))

NoteCropping = collections.namedtuple('NoteCropping', ('pitch', 'start_idx', 'end_idx'))

NoteLabel = collections.namedtuple('NoteLabel', ('instrument_family'))


def timbre_input_tensors_to_model_input(input_tensors):
    """Convert input tensor to FeatureTensors and LabelTensors"""
    spec = tf.reshape(input_tensors['spec'], (-1, constants.TIMBRE_SPEC_BANDS, 1))
    note_croppings = input_tensors['note_croppings']
    instrument_families = input_tensors['instrument_families']

    features = FeatureTensors(
        spec=spec,
        note_croppings=note_croppings,
    )
    labels = LabelTensors(
        instrument_families=instrument_families
    )

    return features, labels


def create_timbre_spectrogram(audio, hparams):
    """Create either a CQT or mel spectrogram"""
    if tf.is_tensor(audio):
        audio = audio.numpy()
    if isinstance(audio, bytes):
        # Get samples from wav data.
        samples = audio_io.wav_data_to_samples(audio, hparams.sample_rate)
    else:
        samples = audio

    if hparams.timbre_spec_type == 'mel':
        spec = np.abs(librosa.feature.melspectrogram(
            samples,
            hparams.sample_rate,
            hop_length=hparams.timbre_hop_length,
            fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
            fmax=librosa.midi_to_hz(constants.MAX_TIMBRE_PITCH),
            n_mels=constants.TIMBRE_SPEC_BANDS,
            pad_mode='symmetric',
            htk=hparams.spec_mel_htk,
            power=2
        )).T

    else:
        spec = np.abs(librosa.core.cqt(
            samples,
            hparams.sample_rate,
            hop_length=hparams.timbre_hop_length,
            fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
            n_bins=constants.TIMBRE_SPEC_BANDS,
            bins_per_octave=constants.BINS_PER_OCTAVE,
            pad_mode='symmetric'
        )).T

    # convert amplitude to power
    if hparams.timbre_spec_log_amplitude:
        spec = librosa.power_to_db(spec) - librosa.power_to_db(np.array([1e-9]))[0]
        spec = spec / np.max(spec)
    return spec


def get_cqt_index(pitch, hparams):
    """Get row closest to this pitch in a CQT spectrogram"""
    frequencies = librosa.cqt_frequencies(constants.TIMBRE_SPEC_BANDS,
                                          fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
                                          bins_per_octave=constants.BINS_PER_OCTAVE)

    return np.abs(frequencies - librosa.midi_to_hz(pitch.numpy() - 1)).argmin()


def get_mel_index(pitch, hparams):
    """Get row closest to this pitch in a mel spectrogram"""
    frequencies = librosa.mel_frequencies(constants.TIMBRE_SPEC_BANDS,
                                          fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
                                          fmax=librosa.midi_to_hz(constants.MAX_TIMBRE_PITCH),
                                          htk=hparams.spec_mel_htk)

    return np.abs(frequencies - librosa.midi_to_hz(pitch.numpy())).argmin()


def include_spectrogram(tensor, hparams=None):
    """Include the spectrogram in our tensor dictionary"""
    spec = tf.py_function(
        functools.partial(create_timbre_spectrogram, hparams=hparams),
        [tensor['audio']],
        tf.float32
    )

    return dict(
        spec=spec,
        note_croppings=tensor['note_croppings'],
        instrument_families=tensor['instrument_families']
    )


def convert_note_cropping_to_sequence_record(tensor, hparams):
    """
    Convert Timbre dataset to be usable by Full Model
    :param note_cropping_dataset: examples are a dicts with keys:
    audio, note_croppings, instrument_families
    :return: same data type as data.parse_example()
    """
    note_croppings = tensor['note_croppings']
    instrument_families = tensor['instrument_families']

    def to_sequence_fn(eager_note_croppings, eager_instrument_families):
        eager_note_croppings = eager_note_croppings.numpy()
        eager_instrument_families = eager_instrument_families.numpy()
        sequence = music_pb2.NoteSequence()
        sequence.tempos.add().qpm = 120
        sequence.ticks_per_quarter = 220
        distinct_families_list = []
        for i in range(len(eager_note_croppings)):
            cropping = NoteCropping(*eager_note_croppings[i])
            family = eager_instrument_families[i].argmax()

            if family not in distinct_families_list:
                distinct_families_list.append(family)

            note = sequence.notes.add()
            note.instrument = distinct_families_list.index(family)
            note.program = instrument_family_mappings.family_to_midi_instrument[family]
            note.start_time = cropping.start_idx / hparams.sample_rate
            note.end_time = cropping.end_idx / hparams.sample_rate
            note.pitch = cropping.pitch
            note.velocity = 70
            if note.end_time > sequence.total_time:
                sequence.total_time = note.end_time
        return sequence.SerializeToString()

    sequence = tf.py_function(to_sequence_fn, [note_croppings, instrument_families], tf.string)

    return dict(
        id='',
        sequence=sequence,
        audio=tensor['audio'],
        velocity_range=music_pb2.VelocityRange(min=0, max=100).SerializeToString()
    )
