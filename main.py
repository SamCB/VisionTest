from __future__ import division, print_function

import importlib
import imp
import sys
import cv2
import argparse

from utils import from_dictionary


def import_module(name):
    if name[-3:] == ".py":
        # assume we're working with a path
        try:
            return imp.load_source("function", name)
        except IOError:
            print("ERROR: Could not find file: {}".format(name))
            print("Exiting")
            sys.exit()
    else:
        # assume we're working with a module
        try:
            return importlib.import_module(name)
        except ImportError:
            print("ERROR: Could not find module: {}".format(name))
            print("Exiting")
            sys.exit()


def main(function, img_input, annotations=None, **kwargs):
    get_answer = import_module(function).initialise()
    camera = import_module(img_input).initialise()
    if annotations:
        get_annotations = import_module(annotations).initialise()
    else:
        get_annotations = None

    show_img = not kwargs.get('silent', False)

    while True:
        img, desc = camera()
        vals = get_answer(img)
        for name, points in vals:
            from_dictionary(points, name=name).draw(img)

        if show_img:
            cv2.imshow(desc, img)

            if cv2.waitKey(1) == 27:
                break

    if show_img:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    description = """\
Test out vision functions for Robot Soccer.
"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-s', '--silent', action='store_true',
        help='do not display images during testing')
    parser.add_argument(
        "function", help="module containing method for performing CV analysis"
    )
    parser.add_argument(
        "input", help="module that provides image frames for the analysis"
    )
    parser.add_argument(
        "annotations", nargs="?", default=None,
        help="module returning the correct annotations for given images"
    )
    args = parser.parse_args()
    main(args.function, args.input, args.annotations, silent=args.silent)
