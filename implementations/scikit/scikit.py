"""Function that uses scikit to predict outputs

Expected arguments:
    algorithm - the name of the algorithm to use for classification
    feature_processor_module - the name of the module that will help process
        images to retrieve features
    data_folder - the folder where the raw data for training is located

For example, you could call:

    python main.py implementations/scikit/scikit.py camera.py -f RandomForest\
            -f implementations/scikit/feature_processors/histogram_processor.py\
            -f ../vision_test_files/automated_crop_classifications

This is actually my recommended setup

The learner expects all the data files to be prepended with their classification

For example:

    BALL-something-0-0001.jpg or
    NOTHING-something-else-0-0002.jpg
"""
import os
import errno

import time

import numpy as np

from sklearn import svm, tree, neighbors, ensemble, metrics
from sklearn.externals import joblib

from crop_functions.harris_crop import retrieve_subsections
from crop_functions.subarea_crop import subarea_crop
from crop_functions.colour_crop import colour_crops

from implementations.scikit.data_loader import load_data
from import_module import import_module

def initialise(algorithm, feature_processor_module, data_folder):
    data_processor = import_module(feature_processor_module).feature_processor()
    dataset = load_data(data_folder, data_processor)

    return ScikitImplementation(algorithm, dataset, data_processor).answer


class ScikitImplementation:

    CLASSIFIER_CONSTRUCTORS = {
        "DecisionTree": tree.DecisionTreeClassifier,
        "SVC": svm.SVC,
        "LinearSVC": svm.LinearSVC,
        "KNeighbors": neighbors.KNeighborsClassifier,
        "RandomForest": lambda: ensemble.RandomForestClassifier(n_estimators=20)
    }

    CONFIDENCE_CUTOFF = 0.6

    def __init__(self, algorithm, data_set, data_processor, min_confidence=0.6):
        self.data = data_set
        self.data_processor = data_processor

        self.min_confidence = min_confidence

        # load_classifier checks if we've saved the classifier to file, if
        #  we have, sweet, we've saved a lot of time.
        self.classifier = self.load_classifier(algorithm, data_processor)

        if self.classifier is None:
            self.classifier = self.create_classifier(algorithm)
            self.classifier.fit(self.data.data, self.data.labels)
            self.save_classifier(algorithm, data_processor, self.classifier)

    def answer(self, image):
        def subsections(image):
            for x, y, w, h in subarea_crop(retrieve_subsections(image)):
            # for x, y, w, h in colour_crops(image):
                yield self.data_processor(image[y:y+h,x:x+w]), (x, y, w, h)

        processed_subsections = []
        crop_areas = []
        for img_subsection, (x, y, w, h) in subsections(image):
            processed_subsections.append(img_subsection)
            crop_areas.append({"x": x, "y": y, "width": w, "height": h})

        return zip(self._classify_subsections(processed_subsections), crop_areas)

    def _classify_subsections(self, crops):
        if len(crops) == 0:
            return []
        if hasattr(self.classifier, "predict_proba"):
            start = time.clock()
            probabilities = self.classifier.predict_proba(crops)
            # Add an extra column to probabilities with our minimum required
            #  confidence and an extra 'nothing' column to the classes, then
            #  return the respective class that is greatest in each row.
            min_column = np.full((len(crops), 1), self.min_confidence)
            min_label = ['nothing']

            limited_probabilities = np.concatenate(
                                        (probabilities, min_column), axis=1)
            limited_classes = np.concatenate(
                                        (self.classifier.classes_, min_label))
            print "Classification time:", time.clock() - start, len(crops)

            return limited_classes[np.argmax(limited_probabilities, axis=1)]
        else:
            return self.classifier.predict(crops)

    @staticmethod
    def classifier_save_name(algorithm_name, data_processor):
        return "./saved_classifier/{}-{}.pkl".format(algorithm_name, data_processor.__name__)

    @staticmethod
    def load_classifier(algorithm_name, data_processor):
        classifier_name = ScikitImplementation.classifier_save_name(algorithm_name, data_processor)
        try:
            # with open(classifier_name, 'r') as f:
            #     return joblib.load(f)
            return joblib.load(classifier_name)
        except IOError:
            return None
        except EOFError:
            return None

    @staticmethod
    def save_classifier(algorithm_name, data_processor, classifier):
        classifier_name = ScikitImplementation.classifier_save_name(algorithm_name, data_processor)
        try:
            os.mkdir(os.path.dirname(classifier_name))
        except OSError as e:
            if not e.errno == errno.EEXIST:
                raise

        # with open(classifier_name, 'w+') as f:
        #     joblib.dump(classifier, f)
        joblib.dump(classifier, classifier_name)

    @staticmethod
    def create_classifier(algorithm_name, **kwargs):
        constructors = ScikitImplementation.CLASSIFIER_CONSTRUCTORS
        try:
            return constructors[algorithm_name](**kwargs)
        except KeyError:
            raise ValueError(
                "Cannot find algorithm of name: '{}'. "
                "Valid algorithms: {}".format(
                    algorithm_name,
                    list(ScikitImplementation.CLASSIFIER_CONSTRUCTORS))
            )
