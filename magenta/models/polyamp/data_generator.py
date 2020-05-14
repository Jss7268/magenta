# Copyright 2020 A. Anonymous. - The PolyAMP Authors.
#
# We have removed the license from our anonymized code.
# All rights reserved.

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
                X, y = ([t.numpy() for t in tensors] for tensors in next(self.iterator))
            else:
                X, y = ([t for t in tensors] for tensors in next(self.iterator))
            return X, y

    def get(self):
        """Simple get item API"""
        return self.__getitem__(0)
