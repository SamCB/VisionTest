"""Return the same general annotation every time"""

def initialise():
    return answer


def answer(img):
    test_annotations = [
        {
            "class": "Ball",
            "height": 66.0,
            "width": 64.0,
            "x": 502.0,
            "y": 267.0
        },
        {
            "class": "Goal Post",
            "height": 168.0,
            "width": 26.0,
            "x": 417.0,
            "y": -4.0
        },
        {
            "class": "Field Line",
            "x1": 272.0,
            "x2": 0.0,
            "y1": 120.0,
            "y2": 130.0
        },
        {
            "class": "Field Line",
            "x1": 270.0,
            "x2": 638.0,
            "y1": 117.0,
            "y2": 212.0
        },
        {
            "class": "Field Line",
            "x1": 397.0,
            "x2": 286.0,
            "y1": 150.0,
            "y2": 159.0
        },
        {
            "class": "Field Line",
            "x1": 285.0,
            "x2": 638.0,
            "y1": 159.0,
            "y2": 297.0
        },
        {
            "class": "Corner",
            "x": 271.0,
            "y": 120.0
        },
        {
            "class": "Corner",
            "x": 285.0,
            "y": 161.0
        },
        {
            "class": "T Junction",
            "x": 393.0,
            "y": 150.0
        },
        {
            "class": "X Junction",
            "x": 273.0,
            "y": 68.0
        },
        {
            "class": "Penalty Spot",
            "x": 166.0,
            "y": 220.0
        },
        {
            "class": "Center Circle",
            "x": 240.0,
            "y": 78.0
        },
        {
            "class": "Nao",
            "height": 86.0,
            "width": 49.0,
            "x": 49.0,
            "y": 12.0
        },
        {
            "class": "Not Ball",
            "height": 38.0,
            "width": 43.0,
            "x": 110.0,
            "y": 19.0
        }
    ]
    return ((a.pop('class'), a) for a in test_annotations)
