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
from video import VideoInput

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


def main(function, function_args,
         img_input, input_args,
         annotations=None, annotation_args=None, kwargs=None):
    get_answer = import_module(function).initialise(*function_args)
    camera = import_module(img_input).initialise(*input_args)
    if annotations:
        get_annotations = import_module(annotations).initialise(*annotation_args)
    else:
        get_annotations = None
    print("Loaded:")
    print("- Function:", function)
    print("- Image Source:", img_input)
    print("- Annotations:", annotations)

    show_img = not kwargs.get('silent', False)
    save_img = kwargs.get('save', False)
    im_count = 1

    while True:
        # Retrieve image and description from our image input
        response = camera()
        if response is None:
            print("Out of Images")
            print("End")
            break
        else:
            img, desc = response

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
            print(comparison_string(comparison=comparison))

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

if __name__ == '__main__':
    description = """\
Test out vision functions for Robot Soccer.
"""
    epilog = """\
---------------------------
Examples:
---------------------------

To run some example function through the
camera input:

    python {name} function.py camera.py

To run some random function with an example
input and the example annotation test:

    python {name} function.py other_input.py annotation.py

If a camera requires input, for example,
the name of video files, you can use:

    python {name} function.py video.py annotation.py -i foo.mp4 -i bar.mp4 -i baz.mp4

Where each -i argument will be handed as a
seperate argument to the initialiser in
video.py.

Similarly, use -f to hand arguments into
the function initialiser and -a to hand
arguments into the annotation initialiser
""".format(name=__file__)
    parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-s', '--silent', action='store_true',
        help='do not display images during testing')
    parser.add_argument(
        '-v', '--save', dest='save', action='store_true',
        help='save cropped images during testing')
    function_group = parser.add_argument_group('function group')
    function_group.add_argument(
        "function", help="module containing method for performing CV analysis"
    )
    function_group.add_argument(
        "-f", "--farg", action="append", default=[],
        help="arguments to pass to the function module initialiser"
    )

    input_group = parser.add_argument_group('input group')
    input_group.add_argument(
        "input", help="module that provides image frames for the analysis"
    )
    input_group.add_argument(
        "-i", "--iarg", action="append", default=[],
        help="arguments to pass to the input module initialiser"
    )

    annotations_group = parser.add_argument_group('annotations group')
    annotations_group.add_argument(
        "annotations", nargs="?", default=None,
        help="module returning the correct annotations for given images"
    )
    annotations_group.add_argument(
        "-a", "--aarg", action="append", default=[],
        help="arguments to pass to the annotations module initialiser"
    )
    args = parser.parse_args()
    
    main(args.function, args.farg,
         args.input, args.iarg,
         args.annotations, args.aarg,
         vars(args))
