from __future__ import division, print_function

import importlib
import imp
import cv2
import argparse

from utils import from_dictionary


def import_module(name):
    if name[-3:] == ".py":
        # assume we're working with a path
        return imp.load_source("function", name)
    else:
        # assume we're working with a module
        return importlib.import_module(name)


def main(function, img_input, **kwargs):
    get_answer = import_module(function).use()
    camera = import_module(img_input).use()

    while True:
        img, desc = camera()
        vals = get_answer(img)
        for name, points in vals:
            from_dictionary(points, name=name).draw(img)

        cv2.imshow(desc, img)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    description = """\
Test out vision functions for Robot Soccer.
"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "function", help="module containing method for performing CV analysis"
    )
    parser.add_argument(
        "input", help="module that provides image frames for the analysis"
    )
    args = parser.parse_args()
    main(args.function, args.input)
