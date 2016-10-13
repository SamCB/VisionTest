# Feature Processors

A feature processor should be written to take an input image (generally something that has already been cropped into an interesting area), and return constant features to classify on. The number and structure of features must ALWAYS be the same, regardless of input image size and shape.

For it to be useable, the module must have the method `feature_processor()` which take no arguments and returns the method object.

Example - Gives every image the features: `[42]`:

```python
def feature_processor():
    return foo

def foo(image):
    return [42]
```
