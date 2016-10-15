from __future__ import print_function
import cv2
import os

from image_utils import resize

def initialise(directory, *args):
    lazy, scale, rate, continue_from = False, 1., 1, 0
    for arg in args:
        if arg == "lazy":
            lazy = True
        elif arg[0] == "s":
            scale = float(arg[1:])
        elif arg[0] == "r":
            rate = int(arg[1:])
        elif arg[0] == "c":
            continue_from = int(arg[1:])

    if lazy:
        return LazyImageSetInput(directory, scale, rate, continue_from).read
    else:
        return ImageSetInput(directory, scale, rate, continue_from).read

class ImageSetInput():

    def __init__(self, directory, scale, rate, continue_from):
        self.images = []
        # if rate == 1, every element, otherwise every 'rate' image
        file_list = os.listdir(directory)[continue_from::rate]
        count = 0
        print("Image settings: Scale: {}, Directory: {}".format(scale, directory))
        for filename in file_list:
            print("Loading Files: {:4}/{:4}".format(count, len(file_list)),
                  end="\r")

            full_path = os.path.join(directory, filename)
            image = cv2.imread(full_path)
            if image is not None:
                rel_path = os.path.relpath(full_path)
                self.images.append((resize(image, scale), rel_path))

            count += 1


        print("Finished Loading {} Images".format(len(self.images)))
        self._index = continue_from
        self.rate = rate

    def read(self):
        try:
            image = self.images[self._index]
        except IndexError:
            return None
        self._index += self.rate
        return image

class LazyImageSetInput():

    def __init__(self, directory, scale, rate, continue_from):
        self.directory = directory
        self.file_list = os.listdir(directory)[continue_from::rate]
        self.scale = scale
        self._index = continue_from
        self.rate = rate
        print("Image settings: Scale: {}, Directory: {}".format(scale, directory))

    def read(self):
        while True:
            try:
                filename = self.file_list[self._index]
            except IndexError:
                return None
            self._index += self.rate

            full_path = os.path.join(self.directory, filename)
            rel_path = os.path.relpath(full_path)
            image = cv2.imread(full_path)
            if image is not None:
                return resize(image, self.scale), rel_path
            # Otherwise, try with the next image
