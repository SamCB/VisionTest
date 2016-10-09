from __future__ import print_function
import cv2
import os


def initialise(directory, lazy=False):
    if lazy:
        return LazyImageSetInput(directory).read
    else:
        return ImageSetInput(directory).read

class ImageSetInput():

    def __init__(self, directory):
        self.images = []
        file_list = os.listdir(directory)
        count = 0
        for filename in file_list:
            print("Loading Files: {:4}/{:4}".format(count, len(file_list)),
                  end="\r")

            full_path = os.path.join(directory, filename)
            image = cv2.imread(full_path)
            if image is not None:
                rel_path = os.path.relpath(full_path)
                self.images.append((image, rel_path))

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

    def __init__(self, directory):
        self.directory = directory
        self.file_list = os.listdir(directory)
        self._index = 0

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
                return image, rel_path
            # Otherwise, try with the next image