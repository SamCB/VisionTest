import json
import argparse

from main import main

if __name__ == '__main__':
    description = """\
Test out vision functions for Robot Soccer.
"""
    epilog = """\
---------------------------
Examples:
---------------------------

Instead of using main.py, save your setup in setup.json like so:

    {{
        "setup_name": {{
            "function": ["function.py", "any", "other", "args"],
            "input": ["input.py", "more arguments"],
            "annotations": ["annotations_no_args.py"]
        }},
        "another_setup": {{
            ...
        }},
        ...
    }}

And call the function with:

    python {name} setup.json setup_name

For quick, easy usage.
""".format(name=__file__)
    parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    #
    # General arguments
    #
    parser.add_argument(
        '-s', '--silent', action='store_true',
        help='do not display images during testing')
    parser.add_argument(
        '-v', '--save', dest='save', action='store_true',
        help='save cropped images during testing')

    # Setup, required arguments
    parser.add_argument(
        "setup_file", help="json file containing setup method we want"
    )
    parser.add_argument(
        "setup_method", help="method in the setup_file to use"
    )

    args = parser.parse_args()

    # Load the settings from the json file
    with open(args.setup_file, "r") as f:
        setup = json.load(f)
    setup_method = setup[args.setup_method]

    # Pull out the attributes we want
    function, fargs = setup_method['function'][0], setup_method['function'][1:]
    img_input, iargs = setup_method['input'][0], setup_method['input'][1:]

    main(function, fargs,
         img_input, iargs,
         silent=args.silent,
         save=args.save)
