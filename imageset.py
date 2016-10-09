from __future__ import print_function
import cv2
import os


def initialise(directory, *args):
    lazy = "lazy" in args
    args = tuple(arg for arg in args if arg != "lazy")

    try:
        scale = float(args[0])
    except ValueError, IndexError:
        scale = 1.

    if lazy:
        return LazyImageSetInput(directory, scale).read
    else:
        return ImageSetInput(directory, scale).read

class ImageSetInput():

    def __init__(self, directory, scale):
        self.images = []
        file_list = os.listdir(directory)
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
        self._index = 0

    def read(self):
        try:
            image = self.images[self._index]
        except IndexError:
            return None
        self._index += 1
        return image

class LazyImageSetInput():

    def __init__(self, directory, scale):
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.scale = scale
        self._index = 0
        print("Image settings: Scale: {}, Directory: {}".format(scale, directory))

    def read(self):
        while True:
            try:
                filename = self.file_list[self._index]
            except IndexError:
                return None
            self._index += 1

            full_path = os.path.join(self.directory, filename)
            rel_path = os.path.relpath(full_path)
            image = cv2.imread(full_path)
            if image is not None:
                return resize(image, self.scale), rel_path
            # Otherwise, try with the next image

def resize(img, scale):
    if scale == 1:
        return img
    else:
        return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA # Assume we're shrinking
        )
