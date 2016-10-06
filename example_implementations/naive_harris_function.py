"""
Naive Harris Function assumes anything that is returned by harris crop is the ball
"""
from crop_functions.harris_crop import retrieve_subsections

def initialise(*args):
    return answer

def answer(img):
    result = []
    for x, y, h, w in retrieve_subsections(img):
        result.append(("ball",
                       {  "height": h,
                          "width": w,
                          "x": x,
                          "y": y
                       }))
    return result
