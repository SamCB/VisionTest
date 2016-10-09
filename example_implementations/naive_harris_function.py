"""
Naive Harris Function assumes anything that is returned by harris crop is the ball

If provided with a true argument, then will include subareas
"""
from crop_functions.harris_crop import retrieve_subsections
from crop_functions.subarea_crop import subarea_crop

def initialise(*args):
    return NaiveHarris(*args).answer

class NaiveHarris:
    def __init__(self, include_subarea=False):
        self.include_subarea = bool(include_subarea)

    def _retrieve_subsections(self, img):
        if self.include_subarea:
            for crop in subarea_crop(retrieve_subsections(img)):
                yield crop
        else:
            for crop in retrieve_subsections(img):
                yield crop

    def answer(self, img):
        result = []
        for x, y, w, h in self._retrieve_subsections(img):
            result.append(("ball",
                           {  "height": h,
                              "width": w,
                              "x": x,
                              "y": y
                           }))
        return result
