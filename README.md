# Robocup Computer Vision Test Suite

This script runs on Python 2 and requires OpenCV 2.4

Tests involving HOG descriptors or the Colour ROI will require the appropriate cpp files to be compiled, and will also require an opencv python/cpp conversion library found at https://github.com/Algomorph/pyboostcvconverter

Usage:

```
python main.py <file/module name for annotation function> <file/module name for image source> <file/module name for the annotation source>
```

For Example, the call:

```
python main.py example_implementations/random_function.py camera.py
```

Will pass images from the webcam to the `answer()` method in the `RandomFunction` class in `random_function.py`. All `answer()` will do is randomly move one of each classification around the image. These images will be output in a popup on the screen.

Or the call:

```
python main.py example_implementations/annotated_function.py example_implementations/annotated_input.py example_implementations/annotated_annotation.py
```

Will mark the response from the return of the function file against the return of the annotation file for each image returned by the input file.

If a module requires initialisation arguments, they can be provided with the `-f`, `-i`, `-a` flags for the function, input and annotation modules respectively. For example:

```
python main.py example_implementations/random_function.py imageset.py -i ../sunny_field_raw/ -i lazy
```

## Using setup file

Instead of using really long command line argument, you can fill out the setup file and use `main_composed.py`.

```json
{
    "setup_name": {
        "function": ["function.py", "any", "other", "args"],
        "input": ["input.py", "more arguments"],
        "annotations": ["annotations_no_args.py"]
    },
    "another_setup": {
        "function": ["something.py"],
        "input": ["input.py", "foo.m4v"]
    }
}
```

```
python main_composed.py setup.json setup_method_name
```

## Writing your own function

We expect a single function `initialise()` which takes no arguments, and returns a method that we can call and pass images into.

An example implementation in a file named `~/some/folder/bad.py` may be:

```python
def initialise():
    # Note, we're returning the method 'answer'
    #  itself, not calling it
    return Bad().answer

class Bad():

    def answer(self, img):
        return [
          ( "Ball",
            { "height": 38.0,
              "width": 43.0,
              "x": 110.0,
              "y": 19.0 }),
          ( "Field Line",
            { "x1": 272.0,
              "x2": 0.0,
              "y1": 120.0,
              "y2": 130.0 }),
          )
        ]
```

And you could call it by

```
python main.py ~/some/folder/bad.py camera.py
```

## Performing Validation

To perform validation using annotated images, you need to have a json file with annotations recorded like so:

```json
[
  { "annotations": [
      {
          "class": "Ball",
          "height": 67.0,
          "width": 63.0,
          "x": 500.0,
          "y": 271.0
      },
      {
          "class": "Nao",
          "height": 55.0,
          "width": 14.0,
          "x": 321.0,
          "y": 73.0
      }
    ],
    "filename": "relative/path/to/image.jpg"
  },
  {},
  {},
  {}
]
```

Then create a new setup in the `setup.json` file which uses `imageset_validator.py` for the input like so:

```json
{ "annotated": {
    "function": [
      "some/function/implementation.py"
    ],
    "input": [
      "imageset_validator.py",
      "../an/example/annotationsManual.json",
      "../yet/another/annotationsManual.json",
      "../../as/many/as/you/want/annotationsManual.json",
      "then some other args like",
      "lazy",
      "s0.5"
    ]
  }
}
```
