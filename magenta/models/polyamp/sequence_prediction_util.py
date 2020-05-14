# Copyright 2020 A. Anonymous. - The PolyAMP Authors.
#
# We have removed the license from our anonymized code.
# All rights reserved.

"""Sequence prediction utilities."""

import tensorflow.keras.backend as K

from magenta.models.polyamp import dataset_reader, instrument_family_mappings
from magenta.music import constants, sequences_lib


def predict_multi_sequence(frame_predictions, onset_predictions=None,
                           offset_predictions=None, active_onsets=None,
                           velocity_values=None, min_pitch=0,
                           hparams=None, qpm=None):
    """Predict NoteSequence from multi-instrument pianoroll."""
    permuted_frame_predictions = K.permute_dimensions(frame_predictions, (2, 0, 1))

    if onset_predictions is not None:
        permuted_onset_predictions = K.permute_dimensions(onset_predictions, (2, 0, 1))
    else:
        permuted_onset_predictions = [None for _ in
                                      range(K.int_shape(permuted_frame_predictions)[0])]

    if offset_predictions is not None:
        permuted_offset_predictions = K.permute_dimensions(offset_predictions, (2, 0, 1))
    else:
        permuted_offset_predictions = [None for _ in
                                       range(K.int_shape(permuted_frame_predictions)[0])]

    if active_onsets is not None:
        permuted_active_onsets = K.permute_dimensions(active_onsets, (2, 0, 1))
    else:
        permuted_active_onsets = permuted_onset_predictions

    multi_sequence = None
    for instrument_idx in range(hparams.timbre_num_classes):
        frame_predictions = permuted_frame_predictions[instrument_idx]
        onset_predictions = permuted_onset_predictions[instrument_idx]
        offset_predictions = permuted_offset_predictions[instrument_idx]
        active_onsets = permuted_active_onsets[instrument_idx]
        sequence = predict_sequence(
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            velocity_values=velocity_values,
            min_pitch=min_pitch,
            hparams=hparams,
            instrument=instrument_idx,
            program=instrument_family_mappings.family_to_midi_instrument[instrument_idx] - 1,
            active_onsets=active_onsets,
            qpm=qpm)
        if multi_sequence is None:
            multi_sequence = sequence
        else:
            multi_sequence.notes.extend(sequence.notes)
    return multi_sequence


def predict_sequence(frame_predictions, onset_predictions=None,
                     offset_predictions=None, active_onsets=None,
                     velocity_values=None, min_pitch=0,
                     hparams=None, qpm=None,
                     instrument=0, program=0):
    """Predict NoteSequence from instrument-agnostic pianoroll."""
    if active_onsets is None:
        # This allows us to set a higher threshold for onsets that we
        # force-add to the frames as opposed to onsets
        # that determine the start of a note.
        active_onsets = onset_predictions

    if qpm is None:
        qpm = constants.DEFAULT_QUARTERS_PER_MINUTE

    if not hparams.predict_onset_threshold:
        onset_predictions = None
    if not hparams.predict_offset_threshold:
        offset_predictions = None
    if not hparams.active_onset_threshold:
        active_onsets = None

    sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
        frames=frame_predictions,
        frames_per_second=dataset_reader.hparams_frames_per_second(hparams),
        min_duration_ms=0,
        min_midi_pitch=min_pitch,
        onset_predictions=onset_predictions,
        offset_predictions=offset_predictions,
        velocity_values=velocity_values,
        instrument=instrument,
        program=program,
        qpm=qpm,
        active_onsets=active_onsets)

    return sequence_prediction
