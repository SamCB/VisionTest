COMPARISON_CLASSES = {"ball", "nao"}
PREDICTION_CLASSES = {"ball", "nao", "ball_part", "nao_part"}

MIN_REQUIRED_PARTIALS = 2

def convert_annotation_class_name(name):
    return name.lower()

def get_prediction_class_name(prediction):
    return prediction[0]

def convert_partial_name(name):
    return "_".join(name.split("_")[:-1])

class ComparisonResults:

    def __init__(self):
        self.predictions = []
        self.ground_truth = []

    def add_comparison(self, predictions, annotations):
        comparison_list = self._create_comparison_lists[annotations]
        false_positive_list = []
        for prediction in predictions:
            prediction_class = get_prediction_class_name(prediction)
            if prediction_class in COMPARISON_CLASSES:
                self._match(prediction, comparison_list, false_positive_list)

        self._apply_comparison(comparison_list, false_positive_list)

    def _apply_comparison(self, comparison_list, false_positive_list):
        for comparison in comparison_list:
            if len(comparison["matched"]) == 0:
                self.predictions.append("nothing")
                self.ground_truth.append(comparison["class"])
            else:
                partials = {}
                for match in comparison["matched"]:
                    if match in COMPARISON_CLASSES:
                        self.predictions.append(match)
                        self.ground_truth.append(comparison["class"])
                    else:
                        # We need to make sure we have enough of the partials
                        #  before appending them as a single case.
                        partials[match] = partials.get(match, 0) + 1

                for partial_name, value in partials.items():
                    if value >= MIN_REQUIRED_PARTIALS:
                        self.predictions.append(convert_partial_name(partial_name))
                        self.ground_truth.append(comparison["class"])

        partials = {}
        for false_positive in false_positive_list:
            if false_positive in COMPARISON_CLASSES:
                self.predictions.append(false_positive)
                self.ground_truth.append("nothing")
            else:
                partials[false_positive] = partials.get(false_positive, 0) + 1

        # This isn't entirely accurate, as it looks for all partials spread
        #  all across the image, not just a lot of partials together which
        #  would be considered an object. Still, it does give us some
        #  information as to reliability
        for partial_name, value in partials.items():
            if value >= MIN_REQUIRED_PARTIALS:
                self.predictions.append(convert_partial_name(partial_name))
                self.ground_truth.append("nothing")

    @staticmethod
    def _create_comparison_lists(annotations):
        comparison_list = []
        for annotation in annotations:
            class_ = convert_class_name(annotation['class'])
            if class_ in COMPARISON_CLASSES:
                comparison_list.append(
                    {"class": class_,
                     "coords": ((annotation['x'], annotation['y']),
                                (annotation['width'], annotation['height']))
                     "matched": []
                    })
        return comparison_list

    @staticmethod
    def _match(prediction, comparison_list, false_positive_list):
        found = False


        if not found:
            false_positive_list.append(get_prediction_class_name(prediction))


def is_inside(rect, target):
    rx, ry, rw, rh = rect
    rx2, ry2 = rx + rw, ry + rh
    tx, ty, tw, th = target
    tx2, ty2 = tx + tw, ty + th
    return (inrange(rx, tx, tx2, l=0.1) and
            inrange(ry, ty, ty2, l=0.1) and
            inrange(rx2, tx, tx2, l=0.1) and
            inrange(ry2, ty, ty2, l=0.1))

def is_similar(rect, target):
    rx, ry, rw, rh = rect
    rx2, ry2 = rx + rw, ry + rh
    tx, ty, tw, th = target
    tx2, ty2 = tx + tw, ty + th

    # How much they are allowed to differ. It might be worth changing this so
    #  it's not linear. (We should be comparatively more lenient on smaller boxes)
    legal_diffs = (rw * 0.1, rh * 0.1, rw * 0.1, rh * 0.1)
    rect_bounding_box = (min(rx, rx2), min(ry, ry2), max(rx, rx2), max(ry, ry2))
    target_bounding_box = (min(tx, tx2), min(ty, ty2), max(tx, tx2), max(ty, ry2))

    for pointA, pointB, legal_diff in zip(rect_bounding_box, target_bounding_box, legal_diffs):
        if abs(pointA - pointB) > legal_diff:
            return False

    return True

def inrange(val, a, b, l=0.):
    return (a - (a * l) <= val <= b + (b * l) or
            b - (b * l) <= val <= a + (a * l))
