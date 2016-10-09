from sklearn import neighbors, tree, svm, naive_bayes
import cv2

from utils import DataSet
from image_utils import get_histogram, image_resize_inscale
from crop_functions.harris_crop import retrieve_subsections
from crop_functions.subarea_crop import subarea_crop
TRAINING_PROPORTION = 1

BALL_FILES_1 = "../vision_test_files/inside_ball_cropped/"
BALL_FILES_2 = "../vision_test_files/outside_ball_cropped/"
BALL_FILES_3 = "../vision_test_files/SPQR/cropped/Ball/"
NOTHING_FILES = "../vision_test_files/robot_ball_fp/"
NAO_FILES_1 = "../vision_test_files/MultiNaoMovingCamera/cropped/Nao/"
NAO_FILES_2 = "../vision_test_files/SPQR/cropped/Nao/"

ALGORITHMS = {"tree": tree.DecisionTreeClassifier,
              "kNeighbors": neighbors.KNeighborsClassifier,
              "svm": svm.LinearSVC,
              "naiveBayes": naive_bayes.GaussianNB}


def initialise(algorithm="tree"):
    try:
        return ScikitLearnt(ALGORITHMS[algorithm]).answer
    except KeyError:
        raise ValueError("Invalid Algorithm: '{}'. "
                         "Valid algorithms are: {}"
                         .format(algorithm, list(ALGORITHMS))
        )


class ScikitLearnt:
    def __init__(self, classifier):
        print "Loading Data"
        self.data = DataSet()
        self.data.add_images_from_folder("ball", BALL_FILES_1)
        self.data.add_images_from_folder("ball", BALL_FILES_2)
        self.data.add_images_from_folder("ball", BALL_FILES_3)
        self.data.add_images_from_folder("none", NOTHING_FILES)
        self.data.add_images_from_folder("nao", NAO_FILES_1)
        self.data.add_images_from_folder("nao", NAO_FILES_2)


        print "Processing Data"
        self.data.confirm(image_scale=(8, 8), histogram_scale=8)

        print "Training Classifier"
        self.classifier = classifier()
        self.training_size = int(len(self.data)*TRAINING_PROPORTION)
        self.classifier.fit(self.data.images[:self.training_size],
                            self.data.labels[:self.training_size])
        print "Finished Training"


    def answer(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subsection_coords = []
        subsection_images = []
        for x, y, w, h in subarea_crop(retrieve_subsections(img)):
            if abs(w - h) > max(w, h)/2:
                # if it's too rectangular, we don't want the image
                continue

            subsection_coords.append((x, y, w, h))
            subsection_images.append(process_subsection(img[y:y+h, x:x+w]))

        if len(subsection_images) == 0:
            return []

        prediction = self.classifier.predict(subsection_images)
        output = []
        for p, (x, y, w, h) in zip(prediction, subsection_coords):
            rect = {"x": x, "y": y, "height": h, "width": w}
            if p == "ball":
                output.append(("Ball", rect))
            if p == "nao":
                output.append(("Nao", rect))
        return output

def process_subsection(subsection):
    return image_resize_inscale(subsection, (8,8))
