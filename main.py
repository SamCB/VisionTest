from __future__ import division, print_function

import importlib
import imp
import sys
import argparse
from pprint import pprint
import os

import numpy as np
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
    save_img = kwargs.get('save', False)
    im_count = 1

    if get_annotations:
        comparison_set = ComparisonSet()

    while True:
        # Retrieve image and description from our image input
        img, desc = camera()

        # Retrieve estimation from our function
        results = get_answer(img)

        # Compare our estimation if we're expecting it
        if get_annotations:
            annotation = get_annotations(desc)
            if annotation is None:
                print("Couldn't find annotation for:", desc)
                print("End")
                break
            comparison = compare_results_to_annotation(results, annotation)
            comparison_set.add(comparison)

        # If we want to save the cropped images, save them.
        if save_img:
            for name, points in results:
                cropped = img[points['y']:points['y']+points['height'], 
                                        points['x']:points['x']+points['width']]
                cv2.imwrite(os.path.join('cropped_ims', 
                                         '0'*(6-int(np.log10(im_count))) +
                                               str(im_count)) + '.bmp', cropped)
                im_count += 1
            
        # If we want to display the image, display it
        if show_img:
            for name, points in results:
                from_dictionary(points, name=name).draw(img)
            cv2.imshow("Test Image", img)

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
        '-v', '--save', dest='save', action='store_true',
        help='save cropped images during testing')
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
    main(args.function, args.input, args.annotations, silent=args.silent, save=args.save)
