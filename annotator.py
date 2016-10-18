from __future__ import print_function

import os
import cv2
import numpy as np

import json

from crop_functions.harris_crop import retrieve_subsections
from crop_functions.subarea_crop import subarea_crop
from crop_functions.colour_crop import ROIFindColour

from import_module import import_module

def main(source, output_folder):
    output_file, output_element_dir = validate_output(output_folder)

    outputs = []

    continue_reading = True

    frame = 0
    while continue_reading:
        img_response = source()
        if img_response is None:
            print("Out of Images")
            print("End")
            break

        img, source_name = img_response
        # crops = ReversableIterator(subarea_crop(ROIFindColour(img)))
        crops = ReversableIterator(retrieve_subsections(img))
        suboutputs = []
        for idx, (x, y, w, h) in crops:
            # if w < 16 or h < 16 or x + w > img.shape[1] or y + h > img.shape[0]:
            if (w > 16 and h > 16) and (w < 8 or h < 8):
                # we want small images right now
                crops.remove_current()
                continue
            try:
                classification = ask_for_class(img, x, y, w, h)
            except KeyboardInterrupt:
                continue_reading = False
                break

            if classification is None:
                crops.back()
                print("BACKSPACE:")
            elif classification == "__SKIP__":
                print("{:4d} - Skipped".format(idx))
            else:
                print("{:4d} - {}".format(idx, classification))

                # Write to the array we'll store in the JSON file
                classification_output = {"class": classification,
                                         "x": x, "y": y, "w": w, "h": h}
                try:
                    suboutputs[idx] = classification_output
                except IndexError:
                    suboutputs.append(classification_output)

        # Write each of the cropped images to file
        for idx, classification in enumerate(suboutputs):
            x = classification["x"]
            y = classification["y"]
            w = classification["w"]
            h = classification["h"]
            subimg = img[y:y+h, x:x+w]
            filename = output_crop_name(output_element_dir, frame, idx, source_name, classification['class'])
            i = 0
            while os.path.isfile(filename):
                filename = output_crop_name(output_element_dir, frame, idx, source_name, classification['class'], str(i))
                i += 1
            cv2.imwrite(filename, subimg)
            classification["filename"] = filename
            classification["sourcename"] = source_name
            classification["frame"] = frame

        # Resave all coordinates to JSON
        outputs.extend(suboutputs)
        with open(output_file, "w+") as f:
            json.dump(outputs, f)

        frame += 1

class ReversableIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.history = []
        self.index = -1

    def __iter__(self):
        self.iterable = iter(self.iterable)
        return self

    def next(self):
        if self.index + 1 == len(self.history):
            # If this fails, then we've reached the end
            self.history.append(self.iterable.next())

        self.index += 1
        return self.index, self.history[self.index]

    def back(self):
        self.index -= 2
        if self.index < 0:
            self.index = -1

    def remove_current(self):
        if self.index < len(self.history):
            del self.history[self.index]
            self.index -= 1
            if self.index < 0:
                self.index = -1

def ask_for_class(img, x, y, w, h):
    subimg, img = get_images(img, x, y, w, h)
    cv2.imshow('img', img)
    cv2.imshow('subimage', subimg)
    return keyboard_class_input()

KEYBOARD_CLASS_MAP = {
    ord("B"): "BALL",
    ord("b"): "BALL_PART",
    ord("N"): "NAO",
    ord("n"): "NAO_PART",
    ord("T"): "INTERSECTION_T",
    ord("t"): "INTERSECTION_T_PART",
    ord("C"): "INTERSECTION_CORNER",
    ord("c"): "INTERSECTION_CORNER_PART",
    ord("X"): "INTERSECTION_X",
    ord("x"): "INTERSECTION_X_PART",
    ord("l"): "LINE_PART",
    ord("G"): "GOAL_POST",
    ord("g"): "GOAL_PART",
    ord("P"): "PENALTY_SPOT",
    ord("p"): "PENALTY_SPOT_PART",
    ord("K"): "KICKOFF_POINT",
    ord("k"): "KICKOFF_POINT_PART",
    ord(" "): "NOTHING",
    ord("s"): "__SKIP__"
}

KEYBOARD_CLASS_STR = ", ".join("'{}'- {}".format(chr(c), class_) for c, class_ in KEYBOARD_CLASS_MAP.items())
VALID_KEY_STR = "VALID KEYS: {}".format(KEYBOARD_CLASS_STR)
VALID_KEY_STR_CLEAR = " " * len(VALID_KEY_STR)
def keyboard_class_input():
    while True:
        print(VALID_KEY_STR, end="\r")
        key_input = cv2.waitKey()
        print(VALID_KEY_STR_CLEAR, end="\r")
        if key_input == 127:  # Backspace
            return None
        elif key_input == 27: # Esc
            raise KeyboardInterrupt()

        try:
            return KEYBOARD_CLASS_MAP[key_input]
        except KeyError:
            print("ERROR: '{}'-{} is not a valid character".format(chr(key_input), key_input))

def get_images(img, x, y, w, h):
    subimg = img[y:y+h,x:x+w]
    # Resize for easier viewing
    img = resize(img, 0.5)
    # Scaled x, y, w, h
    sx, sy, sw, sh = (int(v*0.5) for v in (x, y, w, h))
    # Draw the rectangle so we know where the image is
    cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
    cv2.putText(img, "x:{} y:{} w:{} h:{}".format(x, y, w, h),
                (sx, sy - 2),
                cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
    return subimg, img

def validate_output(output):
    try_make_dir(output)
    output_file = os.path.join(output, "annotation.json")
    output_dir = os.path.join(output, "elements")
    try_make_dir(output_dir)
    return output_file, output_dir

def output_crop_name(directory, frame, idx, original_file, class_, addition=""):
    original_file_name_only, _ = os.path.splitext(os.path.split(original_file)[1])
    return os.path.join(directory, "{}-{}-{}-{:04d}{}.jpg".format(class_, original_file_name_only, frame, idx, addition))

def try_make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as exception:
        import errno
        if exception.errno != errno.EEXIST:
            raise

def resize(img, scale):
    return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Produce image crops and annotation files from program crops")
    parser.add_argument("input", help="The input function to use")
    parser.add_argument("-i", "--iargs", action="append", default=[], help="Arguments to give to the input function")
    parser.add_argument("output_dir", help="The directory to output everything")
    args = parser.parse_args()

    source = import_module(args.input).initialise(*args.iargs)

    main(source, args.output_dir)
