from __future__ import print_function

import cv2
import numpy as np


class DataSet:
    def __init__(self, processor):
        self._data = []
        self._labels = []
        self._processor = processor

    @property
    def loaded_images(self):
        return bool(self._data)

    def add_image(self, image, label):
        self._data.append(image)
        self._labels.append(label)

    def process(self):
        print("Begin Processing", end="\r")
        processed_data = []
        labels = []

        for i, (image, label) in enumerate(zip(self._data, self._labels)):
            processed_data.append(self._processor(image))

            labels.append(label)

            if i % 100:
                print("Processing images: {}/{} - {:5.2f}%".format(i, len(self._data), i*100./len(self._data)), end="\r")

        print("Randomizing data order.                    ", end="\r")
        p = np.random.permutation(len(self._data))
        self.data = np.array(processed_data)[p]
        self.labels = np.array(labels)[p]

        print("Processing complete.    ")

    def __len__(self):
        if hasattr(self, 'labels'):
            return len(self.labels)
        else:
            return 0
