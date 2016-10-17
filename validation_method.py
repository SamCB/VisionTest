import json


def initialise(annotation_files):
    return Comparisons(annotation_files).add_comparison


class Validations:
    def __init__(self, annotation_files):
        self.annotations = {}
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            for element in annotations:
                if self._is_original_annotation(element):
                    filename, elements = self._convert_original_annotation(element)
                else:
                    filename, elements = self._convert_new_annotation(element)

                if filename not in self.annotations:
                    self.annotations[filename] = []

                self.annotations[filename].extend(ComparisonSet(class_, coords) for class_, coords in elements)

    def validate_elements(self, filename, elements):
        if filename not in self.annotations:
            raise ValueError("Do not know file: '{}'.".format(filename))

        annotations = self.annotations[filename]
        for element in elements:
            for annotation in annotations:
                if annotation.is_equal(element[1]):
                    annotation.add_reference(element[0], element[1])

    @staticmethod
    def _is_original_annotation(element):
        return 'annotations' in element

    @staticmethod
    def _convert_original_annotation(element):
        # { "annotations": [
        #     {   "class": "Ball",
        #         "height": 66.0,
        #         "width": 64.0,
        #         "x": 502.0,
        #         "y": 267.0
        #     }],
        #     "class": "image",
        #     "filename": "img_001.png"
        # }
        annotations = []
        for annotation in element['annotations']:
            if manyin(['height', 'width', 'x', 'y'], annotation):
                coordinates = {
                    "height": annotation['height'],
                    "width": annotation['width'],
                    "x": annotation['x'],
                    "y": annotation['y']
                }
            elif manyin(['x', 'y'], annotation):
                # We want to convert points into rectangles
                coordinates = {
                    "height": 20,
                    "width": 20,
                    "x": annotation['x'],
                    "y": annotation['y']
                }
            else:
                continue

            annotations.append((annotation['class'].lower(), coordinates))

        return (element['filename'], annotations)

    @staticmethod
    def _convert_new_annotation(element):
        # {"h": 20,
        #  "w": 17,
        #  "y": 747,
        #  "x": 1468,
        #  "filename": "../LINE_PART-IMG_2036-0-0000.jpg",
        #  "class": "LINE_PART"
        # }
        return (element['filename'], [(element['class'].lower(),
                                       {'height': element['h'],
                                        'width': element['w'],
                                        'x': element['x'],
                                        'y': element['y']})]
        )


class ComparisonSet:
    """Take a known object and check if a classification references it"""

    def __init__(self, class_, coordinates):
        self.class_ = class_
        self.x = coordinates['x']
        self.y = coordinates['y']
        self.w = coordinates['width']
        self.h = coordinates['height']
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h

        self.references = []

    def add_reference(self, class_, coordinates):
        self.references.append((class_, coordinates))

    def is_equal(self, *args):
        if len(args) == 1:
            x = args[0]['x']
            y = args[0]['y']
            w = manyget(args[0], ['width', 'w'])
            h = manyget(args[0], ['height', 'h'])
        elif len(args) == 4:
            x, y, w, h = args
        else:
            raise ValueError("Must have either 1 dictionary or 4 coordinates")

        return self._is_inside(x, y, w, h) or self._is_similar(x, y, w, h)

    def _is_inside(self, x, y, w, h):
        return (inrange(x, self.x, self.x2, l=0.1) and
                inrange(y, self.y, self.y2, l=0.1) and
                inrange(x + w, self.x, self.x2, l=0.1) and
                inrange(y + h, self.y, self.y2, l=0.1))

    def _is_similar(self, x, y, w, h):
        legal_diffs = [w * 0.1, h * 0.1, w * 0.1, h * 0.1]
        our_bounding_box = [min(self.x, self.x2), min(self.y, self.y2),
                            max(self.x, self.x2), max(self.y, self.y2)]
        other_bounding_box = [min(x, x + w), min(y, y + h),
                              max(y, x + w), max(y, y + h)]

        for pointA, pointB, legal_diff in zip(our_bounding_box, other_bounding_box, legal_diffs):
            if abs(pointA - pointB) > legal_diff:
                return False

        return True

def inrange(val, a, b, l=0.):
    return (a - (a * l) <= val <= b + (b * l) or
            b - (b * l) <= val <= a + (a * l))

def manyin(items, target):
    for item in items:
        if item not in target:
            return False
    return True

def manyget(dictionary, items):
    for item in items:
        try:
            return dictionary[item]
        except KeyError:
            pass
    raise KeyError("None of the items: {} were in the dictionary: {}".format(items, dictionary))

def test():
    test_filename_A = "_test_file_A.json"
    test_filename_B = "_test_file_B.json"
    try:
        test_fileA = [
            {"annotations": [
                { "class": "Field Line Part",
                  "x1": 285.0,
                  "x2": 638.0,
                  "y1": 159.0,
                  "y2": 297.0
                },
                { "class": "Corner",
                  "x": 271.0,
                  "y": 120.0
                },
                { "class": "Ball",
                  "height": 66.0,
                  "width": 64.0,
                  "x": 502.0,
                  "y": 267.0
                }],
             "class": "image",
             "filename": "../vision_test_files/sunny_field_raw/0.jpg"},
            {"annotations": [
                { "class": "T Junction",
                  "x": 100.0,
                  "y": 123.0
                },
                { "class": "Ball",
                  "height": 44.0,
                  "width": 46.0,
                  "x": 300.0,
                  "y": 344.0
                }],
            "class": "image",
            "filename": "../vision_test_files/sunny_field_raw/1.jpg"}]
        test_fileB = [
            {"h": 20,  "w": 20, "y": 120, "x": 271,
             "filename": "../vision_test_files/sunny_field_raw/elements/CORNER-0-0-0000.jpg",
             "sourcename": "../vision_test_files/sunny_field_raw/0.jpg",
             "frame": 0, "class": "CORNER"
            },
            {"h": 66,  "w": 64, "y": 267, "x": 502,
             "filename": "../vision_test_files/sunny_field_raw/elements/BALL-0-0-0000.jpg",
             "sourcename": "../vision_test_files/sunny_field_raw/0.jpg",
             "frame": 0, "class": "BALL"
            },
            {"h": 20,  "w": 20, "y": 123, "x": 100,
             "filename": "../vision_test_files/sunny_field_raw/elements/INTERSECTION_T-1-1-0000.jpg",
             "sourcename": "../vision_test_files/sunny_field_raw/1.jpg",
             "frame": 1, "class": "INTERSECTION_T"
            },
            {"h": 44,  "w": 46, "y": 344, "x": 300,
             "filename": "../vision_test_files/sunny_field_raw/elements/BALL-1-1-0000.jpg",
             "sourcename": "../vision_test_files/sunny_field_raw/1.jpg",
             "frame": 1, "class": "BALL"
            }]

        with open(test_filename_A, 'w+') as f:
            json.dump(test_fileA, f)
        with open(test_filename_B, 'w+') as f:
            json.dump(test_fileB, f)


        print "TEST: create and load validation functions"
        validator = Validations([test_filename_A, test_filename_B])

        print "TEST: add elements which are inside the comparison sets"
        validator.validate_elements(
            "../vision_test_files/sunny_field_raw/0.jpg",
            [("ball_part", {"x": 522, "y": 267, "width": 20, "height": 22}),
             ("ball_part", {"x": 533, "y": 279, "width": 15, "height": 12}),
             ("ball_part", {"x": 518, "y": 263, "width": 5, "height": 15}),
             ("ball_part", {"x": 533, "y": 279, "width": -10, "height": -10}),
             ("corner_part", {"x": 271, "y": 120, "width": 5, "height": 5})]
        )

        print "TEST: check comparison sets for first image"
        image_comparisons = validator.annotations["../vision_test_files/sunny_field_raw/0.jpg"]
        for comparison in image_comparisons:
            if comparison.class_ == "ball":
                assert len(comparison.references) == 4, "should have 4 elements in the ball. Actually: {}.\n{}".format(len(comparison.references), comparison.references)
                for class_, coordinates in comparison.references:
                    assert class_ == "ball_part", "class in ball should be named 'ball_part'. Actually: '{}'".format(class_)
            elif comparison.class_ == "corner":
                assert len(comparison.references) == 1, "should have 1 element in the corner. Actually: {}.\n{}".format(len(comparison.references), comparison.references)
                for class_, coordinates in comparison.references:
                    assert class_ == "corner_part", "class in corner should be named 'corner_part'. Actually: '{}'".format(class_)
            else:
                raise AssertionError("Unknown class: {}".format(comparison.class_))

        print "TEST: add elements that take up entire comparison set"
        # New validator
        validator = Validations([test_filename_A, test_filename_B])
        validator.validate_elements(
            "../vision_test_files/sunny_field_raw/0.jpg",
            [("ball", {"x": 500, "y": 262, "h": 60, "w": 70}),
             ("corner", {"x": 270, "y": 122, "h": 19, "w": 21})]
        )

        print "TEST: check comparison sets for first image"
        image_comparisons = validator.annotations["../vision_test_files/sunny_field_raw/0.jpg"]
        for comparison in image_comparisons:
            if comparison.class_ == "ball":
                assert len(comparison.references) == 1, "should have 1 element for the ball. Actually: {}.\n{}".format(len(comparison.references), comparison.references)
                for class_, coordinates in comparison.references:
                    assert class_ == "ball", "class for ball should be named 'ball'. Actually: '{}'".format(class_)
            elif comparison.class_ == "corner":
                assert len(comparison.references) == 1, "should have 1 element for the corner. Actually: {}.\n{}".format(len(comparison.references), comparison.references)
                for class_, coordinates in comparison.references:
                    assert class_ == "corner", "class for corner should be named 'corner'. Actually: '{}'".format(class_)
            else:
                raise AssertionError("Unknown class: {}".format(comparison.class_))



    except:
        print "==========="
        print "TEST FAILED"
        print "==========="
        raise
    else:
        print "============"
        print "TEST SUCCESS"
        print "============"
    finally:
        import os

        try:
            os.remove(test_filename_A)
        except Exception as e:
            print "Unable to remove file: {}.\n{}".format(test_filename_A, e)

        try:
            os.remove(test_filename_B)
        except Exception as e:
            print "Unable to remove file: {}.\n{}".format(test_filename_B, e)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "t":
        test()
