from __future__ import division, print_function

import importlib
import imp
import sys
import argparse
from pprint import pprint

import cv2

from utils import from_dictionary
from comparison import compare_results_to_annotation, comparison_string
from comparison_set import ComparisonSet

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
    print("Loaded:")
    print("- Function:", function)
    print("- Image Source:", img_input)
    print("- Annotations:", annotations)

    show_img = not kwargs.get('silent', False)

    if get_annotations:
        comparison_set = ComparisonSet()

    while True:
        img, desc = camera()
        results = get_answer(img)
        if get_annotations:
            annotation = get_annotations(desc)
            if annotation is None:
                print("Couldn't find annotation for:", desc)
                print("End")
                break
            comparison = compare_results_to_annotation(results, annotation)
            comparison_set.add(comparison)

        if show_img:
            for name, points in results:
                from_dictionary(points, name=name).draw(img)
            cv2.imshow(desc, img)

            if cv2.waitKey(1) == 27:
                break

    if show_img:
        cv2.destroyAllWindows()
    if get_annotations:
        comparison_set.print_averages()

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
