# Robocup Computer Vision Test Suite

This script runs on Python 2 and requires OpenCV 2.4

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
python main.py example_implementations/random_function.py video.py -i test_video.mp4
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
