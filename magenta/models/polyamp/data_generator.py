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

import threading

import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    """Get examples from a tf Dataset for use with
    Keras training.
    """

    def __init__(self, dataset, batch_size, shuffle=False, use_numpy=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_numpy = use_numpy
        self.iterator = iter(self.dataset)
        self.lock = threading.Lock()
        self.on_epoch_end()

    def __len__(self):
        return 0

    def __getitem__(self, index):
        with self.lock:
            if self.use_numpy:
                x, y = ([t.numpy() for t in tensors] for tensors in next(self.iterator))
            else:
                x, y = ([t for t in tensors] for tensors in next(self.iterator))
            return x, y

    def get(self):
        """Simple get item API"""
        return self.__getitem__(0)
