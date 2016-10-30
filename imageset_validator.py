from __future__ import print_function
import cv2
import os
import json
import numbers

from image_utils import resize

def initialise(*args):
    lazy, scale, rate, continue_from = False, 1., 1, 0
    annotation_files = []
    for arg in args:
        if arg.endswith("json"):
            annotation_files.append(arg)
        elif arg == "lazy":
            lazy = True
        elif arg[0] == "s":
            scale = float(arg[1:])
        elif arg[0] == "r":
            rate = int(arg[1:])
        elif arg[0] == "c":
            continue_from = int(arg[1:])

    if len(annotation_files) == 0:
        raise ValueError("No annotation files found")

    if lazy:
        return ValidatingLazyImageSetInput(annotation_files, scale, rate, continue_from).read
    else:
        return ValidatingImageSetInput(annotation_files, scale, rate, continue_from).read

class ValidatingImageSetInput():
    # No Wait. I don't actually want to inherit. I want to load images my own
    #  way, from the annotation file's perspective
    def __init__(self, annotation_files, scale, rate, continue_from):
        self.annotations = []
        self.images = []
        for group_num, annotation_file in enumerate(annotation_files):
            directory = os.path.dirname(annotation_file)
            with open(annotation_file, "r") as f:
                loaded_annotations = json.load(f)

            for img_num, image_annotation in enumerate(loaded_annotations):
                print("Loading Files: {:4}/{:4} in group {:2}/{:2}".format(
                        img_num, len(loaded_annotations),
                        group_num + 1, len(annotation_files)),
                      end="\r")

                full_path = os.path.join(directory, image_annotation['filename'])
                image = cv2.imread(full_path)
                if image is not None:
                    self.annotations.append(scale_annotation(image_annotation['annotations'], scale))
                    self.images.append(resize(image, scale))
                else:
                    print("WARNING: could not find or load '{}'".format(full_path))

        print("Finished loading {} image(s) from {} group(s)".format(
                len(self.images), len(annotation_files)))
        self._index = continue_from
        self.rate = rate

    def read(self):
        try:
            image = self.images[self._index]
            annotation = self.annotations[self._index]
        except IndexError:
            return None

        self._index += self.rate

        return image, annotation

class ValidatingLazyImageSetInput():

    def __init__(self, annotation_files, scale, rate, continue_from):
        self.image_annotations = []
        for annotation_file in annotation_files:
            directory = os.path.dirname(annotation_file)
            with open(annotation_file, "r") as f:
                loaded_annotations = json.load(f)

            for image_annotation in loaded_annotations:
                self.image_annotations.append((directory, image_annotation))

        self.scale = scale
        self._index = continue_from
        self.rate = rate

    def read(self):
        while True:
            try:
                directory, image_annotation = self.image_annotations[self._index]
            except IndexError:
                return None
            self._index += self.rate

            full_path = os.path.join(directory, image_annotation['filename'])
            image = cv2.imread(full_path)
            if image is not None:
                return resize(image, self.scale), scale_annotation(image_annotation['annotations'], self.scale)
            # Otherwise, try with the next image

def scale_annotation(annotations, scale):
    for annotation in annotations:
        for key, val in annotation.items():
            if isinstance(val, numbers.Number):
                annotation[key] = val * scale
    return annotations
