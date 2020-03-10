# load and save models
import glob
import os
import time
from enum import Enum

import librosa.display
import numpy as np
import uuid
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from sklearn.utils import class_weight
from tensorflow.keras.layers import BatchNormalization

from magenta.models.onsets_frames_transcription import infer_util, constants
from magenta.models.onsets_frames_transcription.callback import MidiPredictionMetrics, \
    TimbrePredictionMetrics

import tensorflow.compat.v1 as tf

from magenta.models.onsets_frames_transcription.layer_util import get_croppings_for_single_image
from magenta.models.onsets_frames_transcription.loss_util import log_loss_wrapper
from magenta.models.onsets_frames_transcription.nsynth_reader import NoteCropping
from magenta.models.onsets_frames_transcription.timbre_model import timbre_prediction_model, \
    acoustic_model_layer

FLAGS = tf.app.flags.FLAGS

if FLAGS.using_plaidml:
    # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    import plaidml.keras

    plaidml.keras.install_backend()
    from keras.models import load_model
    from keras.optimizers import Adam
    import keras.backend as K
else:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K

from magenta.models.onsets_frames_transcription.data_generator import DataGenerator

from magenta.models.onsets_frames_transcription.accuracy_util import AccuracyMetric, \
    binary_accuracy_wrapper
from magenta.models.onsets_frames_transcription.midi_model import midi_prediction_model


class ModelType(Enum):
    MIDI = 'Midi',
    TIMBRE = 'Timbre',
    FULL = 'Full',


def to_lists(x, y):
    return tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)


class ModelWrapper:
    def __init__(self, model_dir, type, id=None, batch_size=8, steps_per_epoch=100,
                 dataset=None,
                 accuracy_metric=AccuracyMetric('acc', 'accuracy'),
                 model=None, hist=None, hparams=None):
        self.model_dir = model_dir
        self.type = type

        self.model_save_format = '{}/Training {} Model Weights {} {:.2f} {:.2f} {}.hdf5' \
            if type is ModelType.MIDI \
            else '{}/Training {} Model Weights {} {:.2f} {}.hdf5'
        self.history_save_format = '{}/Training {} History {} {:.2f} {:.2f} {}' \
            if type is ModelType.MIDI \
            else '{}/Training {} History {} {:.2f} {}.hdf5'
        if id is None:
            self.id = uuid.uuid4().hex
        else:
            self.id = id
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.accuracy_metric = accuracy_metric
        self.model = model
        self.hist = hist
        self.hparams = hparams

        self.dataset = dataset
        if dataset is None:
            self.generator = None
        else:
            self.generator = DataGenerator(self.dataset, self.batch_size, self.steps_per_epoch,
                                           use_numpy=False,
                                           coagulate_mini_batches=type is not ModelType.MIDI and hparams.timbre_coagulate_mini_batches)
        self.metrics = MidiPredictionMetrics(self.generator,
                                             self.hparams) if type == ModelType.MIDI else TimbrePredictionMetrics(
            self.generator, self.hparams)

    def save_model_with_metrics(self, epoch_num):
        if self.type == ModelType.MIDI:
            id_tup = (self.model_dir, self.type.name, self.id,
                      self.metrics.metrics_history[-1].frames['f1_score'].numpy() * 100,
                      self.metrics.metrics_history[-1].onsets['f1_score'].numpy() * 100,
                      epoch_num)
        else:
            id_tup = (self.model_dir, self.type.name, self.id,
                      self.metrics.metrics_history[-1].timbre_prediction['f1_score'].numpy() * 100,
                      epoch_num)
        print('Saving {} model...'.format(self.type.name))
        self.model.save_weights(self.model_save_format.format(*id_tup))
        np.save(self.history_save_format.format(*id_tup), [self.metrics.metrics_history])
        print('Model weights saved at: ' + self.model_save_format.format(*id_tup))

    def train_and_save(self, epochs=1, epoch_num=0):
        if self.model is None:
            self.build_model()

        # self.model.fit_generator(self.generator, steps_per_epoch=self.steps_per_epoch,
        #                          epochs=epochs, workers=2, max_queue_size=8,
        #                          callbacks=[self.metrics])
        # self.model.fit(self.dataset, steps_per_epoch=self.steps_per_epoch,
        #                epochs=epochs, workers=2, max_queue_size=8,
        #                callbacks=[self.metrics])
        for i in range(self.steps_per_epoch):
            x, y = self.generator.get()
            if self.type == ModelType.MIDI:
                class_weights = None  # class_weight.compute_class_weight('balanced', np.unique(y[0]), y[0])
            else:
                class_weights = self.hparams.timbre_class_weights
            # print(np.argmax(y[0], -1))

            # self.plot_spectrograms(x)
            # print('next batch...')
            start = time.perf_counter()
            # new_metrics = self.model.predict(x)
            new_metrics = self.model.train_on_batch(x, y, class_weight=class_weights)
            # self.model.evaluate(x, y)
            print(f'Trained batch {i} in {time.perf_counter() - start:0.4f} seconds')
            print(new_metrics)
        self.metrics.on_epoch_end(1, model=self.model)

        self.save_model_with_metrics(epoch_num)

    def plot_spectrograms(self, x, temporal_ds=16, freq_ds=4, max_batches=1):
        for batch_idx in range(max_batches):
            spec = K.pool2d(x[0], (temporal_ds, freq_ds), (temporal_ds, freq_ds), padding='same')
            croppings = get_croppings_for_single_image(spec[batch_idx], x[1][batch_idx],
                                                       x[2][batch_idx], self.hparams, temporal_ds)
            plt.figure(figsize=(16, 12))
            num_crops = min(3, x[2][batch_idx].numpy())
            plt.subplot(int(num_crops / 2 + 1), 2, 1)
            y_axis = 'cqt_note' if self.hparams.spec_type == 'cqt' else 'mel'
            librosa.display.specshow(librosa.power_to_db(
                tf.transpose(tf.reshape(x[0][batch_idx], x[0][batch_idx].shape[0:-1])).numpy()),
                y_axis=y_axis,
                hop_length=self.hparams.timbre_hop_length,
                fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
                fmax=librosa.midi_to_hz(constants.MAX_TIMBRE_PITCH),
                bins_per_octave=constants.BINS_PER_OCTAVE)
            for i in range(num_crops):
                plt.subplot(int(num_crops / 2 + 1), 2, i + 2)
                db = librosa.power_to_db(
                    tf.transpose(tf.reshape(croppings[i], croppings[i].shape[0:-1])).numpy())
                librosa.display.specshow(db,
                                         y_axis=y_axis,
                                         hop_length=self.hparams.timbre_hop_length,
                                         fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
                                         fmax=librosa.midi_to_hz(constants.MAX_TIMBRE_PITCH),
                                         bins_per_octave=constants.BINS_PER_OCTAVE / freq_ds)
        plt.show()

    def _predict_timbre(self, spec):
        pitch = constants.MIN_TIMBRE_PITCH
        start_idx = 0
        end_idx = self.hparams.timbre_hop_length * spec.shape[1]
        note_croppings = [NoteCropping(pitch=pitch,
                                       start_idx=start_idx,
                                       end_idx=end_idx)]
        note_croppings = tf.reshape(note_croppings, (1, 1, 3))
        num_notes = tf.expand_dims(1, axis=0)

        timbre_probs = self.model.predict([spec, note_croppings, num_notes])
        print(timbre_probs)
        return K.flatten(tf.nn.top_k(timbre_probs).indices)

    def _predict_sequence(self, spec):
        y_pred = self.model.predict(spec)
        frame_predictions = y_pred[0][0] > self.hparams.predict_frame_threshold
        onset_predictions = y_pred[1][0] > self.hparams.predict_onset_threshold
        offset_predictions = y_pred[2][0] > self.hparams.predict_offset_threshold

        # frame_predictions = tf.expand_dims(frame_predictions, axis=0)
        # onset_predictions = tf.expand_dims(onset_predictions, axis=0)
        # offset_predictions = tf.expand_dims(offset_predictions, axis=0)
        sequence = infer_util.predict_sequence(
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            velocity_values=None,
            hparams=self.hparams, min_pitch=constants.MIN_MIDI_PITCH)
        return sequence

    def predict_from_spec(self, spec):
        if self.type == ModelType.MIDI:
            return self._predict_sequence(spec)
        elif self.type == ModelType.TIMBRE:
            return self._predict_timbre(spec)

    def load_newest(self):
        try:
            model_weights = \
            sorted(glob.glob(f'{self.model_dir}/Training {self.type.name} Model Weights *.hdf5'),
                   key=os.path.getmtime)[-1]
            model_history = \
            sorted(glob.glob(f'{self.model_dir}/Training {self.type.name} History *.npy'),
                   key=os.path.getmtime)[-1]
            self.metrics.load_metrics(
                np.load(model_history,
                        allow_pickle=True)[0])
            print('Loading pre-trained model: {}'.format(model_weights))
            self.model.load_weights(model_weights)
            print('Model loaded successfully')
        except:
            print(f'Couldn\'t load model weights')

    def load_model(self, frames_f1, onsets_f1=-1, id=-1, epoch_num=0):
        if not id:
            id = self.id
        self.build_model()
        if self.type == ModelType.MIDI:
            id_tup = (self.model_dir, self.type.name, id, frames_f1, onsets_f1, epoch_num)
        else:
            id_tup = (self.model_dir, self.type.name, id, frames_f1, epoch_num)
        if os.path.exists(self.model_save_format.format(*id_tup)) \
                and os.path.exists(self.history_save_format.format(*id_tup) + '.npy'):
            try:
                self.metrics.load_metrics(
                    np.load(self.history_save_format.format(*id_tup) + '.npy',
                            allow_pickle=True)[0])
                print('Loading pre-trained {} model...'.format(self.type.name))
                self.model.load_weights(self.model_save_format.format(*id_tup))
                print('Model loaded successfully')
            except:
                print(f'Couldn\'t load model weights')
        else:
            print('Couldn\'t find pre-trained model: {}'
                  .format(self.model_save_format.format(*id_tup)))

    # num_classes only needed for timbre prediction
    def build_model(self):
        if self.type == ModelType.MIDI:
            self.model, losses, accuracies = midi_prediction_model(self.hparams)
            self.model.compile(Adam(self.hparams.learning_rate,
                                    decay=self.hparams.decay_rate,
                                    clipnorm=self.hparams.clip_norm),
                               metrics=accuracies, loss=losses)
        elif self.type == ModelType.TIMBRE:
            self.model, losses, accuracies = timbre_prediction_model(self.hparams)
            self.model.compile(Adam(self.hparams.timbre_learning_rate,
                                    decay=self.hparams.timbre_decay_rate,
                                    clipnorm=self.hparams.timbre_clip_norm),
                               metrics=accuracies, loss=losses)
        else:  # self.type == ModelType.FULL:
            pass

        print(self.model.summary())
