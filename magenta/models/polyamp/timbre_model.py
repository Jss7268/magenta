# Copyright 2020 Jack Smith.
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

import functools
import math

from dotmap import DotMap
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, Conv1D, Conv2D, ConvLSTM2D, Dense, Dropout, \
    ELU, Flatten, GlobalMaxPooling1D, Input, Lambda, MaxPooling1D, Reshape, SpatialDropout2D, \
    TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from magenta.models.polyamp import constants
from magenta.models.polyamp.accuracy_util import flatten_accuracy_wrapper, flatten_f1_wrapper
from magenta.models.polyamp.layer_util import conv_elu_pool_layer, \
    get_all_croppings
from magenta.models.polyamp.loss_util import timbre_loss_wrapper


def get_default_hparams():
    return {
        'timbre_learning_rate': 4e-4,
        'timbre_decay_steps': 10000,
        'timbre_decay_rate': 1e-2,
        'timbre_clip_norm': 2.0,
        'timbre_l2_regularizer': 1e-7,
        'timbre_filter_frequency_sizes': [3, 7],
        'timbre_filter_temporal_sizes': [1, 3, 5],
        'timbre_num_filters': [96, 64, 32],
        'timbre_filters_pool_size': (int(64 / 4), int(constants.BINS_PER_OCTAVE / 14)),
        'timbre_pool_size': [(3, 2), (3, 2)],
        'timbre_conv_num_layers': 2,
        'timbre_dropout_drop_amts': [0.1, 0.1, 0.1],
        'timbre_rnn_dropout_drop_amt': [0.3, 0.5],
        'timbre_fc_size': 512,
        'timbre_penultimate_fc_size': 256,
        'timbre_fc_num_layers': 0,
        'timbre_fc_dropout_drop_amt': 0.0,
        'timbre_conv_drop_amt': 0.1,
        'timbre_final_dropout_amt': 0.0,
        'timbre_local_conv_size': 7,
        'timbre_local_conv_strides': 1,  # reduce memory requirement
        'timbre_local_conv_num_filters': 256,
        'timbre_input_shape': (None, constants.TIMBRE_SPEC_BANDS, 1),  # (None, 229, 1),
        'timbre_num_classes': constants.NUM_INSTRUMENT_FAMILIES + 1,  # include other
        'timbre_lstm_units': 192,
        'timbre_rnn_stack_size': 0,
        'timbre_leaky_alpha': 0.33,
        'timbre_penultimate_activation': 'elu',
        'timbre_final_activation': 'sigmoid',
        'timbre_spatial_dropout': True,
        'timbre_bottleneck_filter_num': 0,
        'timbre_gradient_exp': 12,  # 16 for cqt no-log
        'timbre_spec_epsilon': 1e-8,
    }


class TimbreModel:
    def __init__(self, hparams):
        if hparams is None:
            hparams = DotMap(get_default_hparams())
        self.hparams = hparams

    def acoustic_model_layer(self):
        num_filters = 128

        def acoustic_model_fn(inputs):
            outputs = inputs

            for i in range(self.hparams.timbre_conv_num_layers):
                outputs = conv_elu_pool_layer(
                    num_filters, 3, 3, self.hparams.timbre_pool_size[i],
                    activation_fn=ELU(self.hparams.timbre_leaky_alpha))(outputs)
                if self.hparams.timbre_spatial_dropout:
                    outputs = SpatialDropout2D(self.hparams.timbre_dropout_drop_amts[i])(outputs)
                else:
                    outputs = Dropout(self.hparams.timbre_dropout_drop_amts[i])(outputs)

            return outputs

        return acoustic_model_fn

    def parallel_filters_layer(self):
        # Don't pool yet so we have more cropping accuracy.
        pool_size = (1, 1)

        def filters_layer_fn(inputs):
            parallel_layers = []
            for f_i in self.hparams.timbre_filter_frequency_sizes:
                for i, t_i in enumerate(self.hparams.timbre_filter_temporal_sizes):
                    outputs = conv_elu_pool_layer(self.hparams.timbre_num_filters[i], t_i, f_i,
                                                  pool_size,
                                                  activation_fn=ELU(
                                                      self.hparams.timbre_leaky_alpha))(inputs)
                    if self.hparams.timbre_spatial_dropout:
                        outputs = SpatialDropout2D(self.hparams.timbre_conv_drop_amt)(outputs)
                    else:
                        outputs = Dropout(self.hparams.timbre_conv_drop_amt)(outputs)
                    parallel_layers.append(outputs)
            return K.concatenate(parallel_layers, axis=-1)

        return filters_layer_fn

    def local_conv_layer(self):
        size = self.hparams.timbre_local_conv_size
        strides = self.hparams.timbre_local_conv_strides

        def local_conv_fn(inputs):
            outputs = ELU(self.hparams.timbre_leaky_alpha)(
                TimeDistributed(Conv1D(
                    self.hparams.timbre_local_conv_num_filters,
                    size,
                    strides,
                    padding='same',
                    use_bias=False,
                ), name=f'roi_conv1d_{size}_{strides}')(inputs))

            outputs = TimeDistributed(GlobalMaxPooling1D(), name='global_max_pitch')(outputs)

            return outputs

        return local_conv_fn

    def acoustic_dense_layer(self):
        def acoustic_dense_fn(inputs):
            outputs = inputs
            for i in range(self.hparams.timbre_fc_num_layers):
                outputs = TimeDistributed(Dense(self.hparams.timbre_fc_size,
                                                kernel_initializer='he_uniform',
                                                bias_regularizer=l2(1e-1),
                                                use_bias=False,
                                                activation='sigmoid',
                                                name=f'acoustic_dense_{i}'))(outputs)
                # Don't do batch normalization because our samples
                # are no longer independent.
                outputs = Dropout(self.hparams.timbre_fc_dropout_drop_amt)(outputs)

            penultimate_outputs = TimeDistributed(
                Dense(self.hparams.timbre_penultimate_fc_size,
                      use_bias=True,  # bias so that low bass notes can be predicted
                      bias_regularizer=l2(self.hparams.timbre_l2_regularizer),
                      activation=self.hparams.timbre_penultimate_activation,
                      kernel_initializer='he_uniform'),
                name=f'penultimate_dense_{self.hparams.timbre_local_conv_num_filters}')(outputs)
            return penultimate_outputs

        return acoustic_dense_fn

    def instrument_prediction_layer(self):
        def instrument_prediction_fn(inputs):
            outputs = inputs
            outputs = TimeDistributed(Dense(self.hparams.timbre_num_classes,
                                            activation=self.hparams.timbre_final_activation,
                                            use_bias=False,
                                            kernel_initializer='he_uniform'),
                                      name='timbre_prediction')(outputs)
            return outputs

        return instrument_prediction_fn

    def get_timbre_model(self):
        """Build the Timbre Model architecture."""

        spec = Input(shape=self.hparams.timbre_input_shape, name='spec')

        # batched dimensions for cropping like:
        # ((top_crop, bottom_crop), (left_crop, right_crop))
        # with a high pass, top_crop will always be 0, bottom crop is relative to pitch
        note_croppings = Input(shape=(None, 3),
                               name='note_croppings', dtype='int64')

        spec_with_epsilon = Lambda(lambda x: x + self.hparams.timbre_spec_epsilon)(spec)
        # Acoustic_outputs shape: (None, None, 57, 128).
        # aka: (batch_size, length, freq_range, num_channels)
        acoustic_outputs = self.acoustic_model_layer()(spec_with_epsilon)
        # Filter_outputs shape: (None, None, 57, 448).
        # aka: (batch_size, length, freq_range, num_channels)
        filter_outputs = self.parallel_filters_layer()(acoustic_outputs)
        # Simplify to save memory.
        if self.hparams.timbre_bottleneck_filter_num:
            filter_outputs = Conv2D(self.hparams.timbre_bottleneck_filter_num, (1, 1),
                                    activation='relu')(
                filter_outputs)
        if self.hparams.timbre_rnn_stack_size > 0:
            # Run TimeDistributed LSTM, distributing over pitch.
            # lstm_input = Permute((2, 1, 3))(filter_outputs)
            lstm_input = Lambda(lambda x: K.expand_dims(x, -2))(filter_outputs)
            lstm_outputs = Bidirectional(ConvLSTM2D(
                self.hparams.timbre_lstm_units,
                kernel_size=(5, 1),
                padding='same',
                return_sequences=True,
                dropout=self.hparams.timbre_rnn_dropout_drop_amt[0],
                recurrent_dropout=self.hparams.timbre_rnn_dropout_drop_amt[1],
                kernel_initializer='he_uniform'))(lstm_input)
            # Reshape "does not include batch axis".
            reshaped_outputs = Reshape((-1,
                                        K.int_shape(lstm_outputs)[2] * K.int_shape(lstm_outputs)[3],
                                        K.int_shape(lstm_outputs)[4]))(lstm_outputs)
        else:
            reshaped_outputs = filter_outputs
        # batch_size is excluded from this shape as it gets automatically inferred.
        # output_shape: (batch_size, num_notes, freq_range, num_filters)
        output_shape = (
            None,
            math.ceil(K.int_shape(reshaped_outputs)[2]),
            K.int_shape(reshaped_outputs)[3]
        )
        pooled_outputs = Lambda(
            functools.partial(get_all_croppings, hparams=self.hparams), dynamic=True,
            output_shape=output_shape)(
            [reshaped_outputs, note_croppings])

        # We now need to use TimeDistributed because we have 5 dimensions,
        # and want to operate on thelast 3 independently:
        # (time, frequency, and number of channels/filters).
        pooled_outputs = TimeDistributed(
            MaxPooling1D(pool_size=(self.hparams.timbre_filters_pool_size[1],),
                         padding='same'),
            name='post_crop_pool')(pooled_outputs)

        if self.hparams.timbre_local_conv_num_filters:
            pooled_outputs = self.local_conv_layer()(pooled_outputs)
        # Flatten while preserving batch and time dimensions.
        flattened_outputs = TimeDistributed(Flatten(), name='flatten')(
            pooled_outputs)
        penultimate_outputs = self.acoustic_dense_layer()(flattened_outputs)
        # shape: (None, None, 11)
        # aka: (batch_size, num_notes, num_classes)
        instrument_family_probs = self.instrument_prediction_layer()(penultimate_outputs)

        # Remove padded predictions.
        def remove_padded(input_list):
            probs, croppings = input_list
            end_indices = K.expand_dims(K.permute_dimensions(croppings, (2, 0, 1))[-1], -1)
            # Remove negative end_indices.
            return probs * K.cast_to_floatx(end_indices >= 0)

        instrument_family_probs = Lambda(remove_padded, name='family_probs')(
            [instrument_family_probs, note_croppings])

        if self.hparams.timbre_final_activation == 'sigmoid':
            # This is the only supported option.
            losses = {'family_probs': timbre_loss_wrapper(hparams=self.hparams, recall_weighing=4.)}

        elif self.hparams.timbre_final_activation == 'softmax':
            # TODO use multi-class labelling loss function.
            losses = {'family_probs': timbre_loss_wrapper(hparams=self.hparams, recall_weighing=4.)}

        else:
            # TODO use logit loss function.
            losses = {'family_probs': timbre_loss_wrapper(hparams=self.hparams, recall_weighing=4.)}

        accuracies = {'family_probs': [flatten_accuracy_wrapper(),
                                       lambda *x: flatten_f1_wrapper()(*x)['f1_score']]}

        return Model(inputs=[spec, note_croppings],
                     outputs=instrument_family_probs), losses, accuracies
