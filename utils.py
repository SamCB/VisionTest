import cv2
import numpy as np

from collections import Counter

FONT = cv2.FONT_HERSHEY_PLAIN
COLOUR = (0, 0, 255)
LINE_WIDTH = 1


class Rectangle:
    def __init__(self, x, y, width, height, name=None):
        self.x = np.float32(x)
        self.y = np.float32(y)
        self.height = np.float32(height)
        self.width = np.float32(width)
        self.name = name

    def draw(self, img, colour=COLOUR):
        cv2.rectangle(img,
                      (self.x, self.y),
                      (self.x + self.width, self.y + self.height),
                      colour,
                      LINE_WIDTH)
        if self.name:
            title = "{}: x:{:.1f} y:{:.1f} w:{:.1f} h:{:.1f}".format(
                self.name, self.x, self.y, self.width, self.height
            )
        else:
            title = "x:{:.1f} y:{:.1f} w:{:.1f} h:{:.1f}".format(
                self.x, self.y, self.width, self.height
            )
        cv2.putText(img, title,
                    (self.x, self.y - np.float32(2)),
                    FONT, 0.5, colour, 1)

    def sq_difference(self, other):
        def rotate_points(rect):
            yield rect.x, rect.y, rect.height, rect.width
            yield rect.x + rect.width, rect.y, rect.height, -rect.width
            yield rect.x + rect.width, rect.y + rect.height, -rect.height, -rect.width
            yield rect.x, rect.y + rect.height, -rect.height, rect.width

        def compare_points(a, b):
            return (a.x - b[0])**2 + (a.y - b[1])**2 + (a.height - b[2])**2 + (a.width - b[3])**2

        return min(compare_points(self, o) for o in rotate_points(other))

    def __repr__(self):
        return "{}({}, {}, {}, {}, {})".format(self.__class__.__name__,
                                               self.x, self.y,
                                               self.width, self.height,
                                               self.name)


class Point:

    CIRCLE_RADIUS = 3

    def __init__(self, x, y, name=None):
        self.x = np.float32(x)
        self.y = np.float32(y)
        self.name = name

    def draw(self, img, colour=COLOUR):
        cv2.circle(img, (self.x, self.y), Point.CIRCLE_RADIUS, colour, LINE_WIDTH)
        if self.name:
            title = "{}: x:{:.1f} y:{:.1f}".format(self.name, self.x, self.y)
        else:
            title = "x:{:.1f} y:{:.1f}".format(self.x, self.y)
        cv2.putText(img, title,
                    (self.x, self.y - np.float32(Point.CIRCLE_RADIUS + 1)),
                    FONT, 0.5, colour, 1)

    def sq_difference(self, other):
        return (self.x - other.x)**2 + (self.y - other.y)**2

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__,
                                       self.x, self.y, self.name)


class Line:
    def __init__(self, x1, y1, x2, y2, name=None):
        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)
        self.name = name

    def draw(self, img, colour=COLOUR):
        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), colour, LINE_WIDTH)
        if self.name:
            title = "{}: x1:{:.1f} y1:{:.1f} x2:{:.1f} y2:{:.1f}".format(
                self.name, self.x1, self.y1, self.x2, self.y2
            )
        else:
            title = "x1:{:.1f} y1:{:.1f} x2:{:.1f} y2:{:.1f}".format(
                self.x1, self.y1, self.x2, self.y2
            )
        if self.y1 >= self.y2:
            # slope down, text should be above
            pos = (self.x1, self.y1 - np.float32(1))
        else:
            # slope up, text should be below
            pos = (self.x1, self.y1 + np.float32(6))

    def sq_difference(self, other):
        def rotate_points(line):
            yield line.x1, line.y1, line.x2, line.y2
            yield line.x2, line.y2, line.x1, line.y1

        def compare_points(a, b):
            return (a.x1 - b[0])**2 + (a.y1 - b[1])**2 + (a.x2 - b[2])**2 + (a.y2 - b[3])**2

        return min(compare_points(self, o) for o in rotate_points(other))

    def __repr__(self):
        return "{}({}, {}, {}, {})".format(self.__class__.__name__,
                                           self.x1, self.y1,
                                           self.x2, self.y2)


def from_dictionary(dict_, name=None):
    if name is None:
        # If name not provided, maybe it's provided in the dictionary
        name = dict_.pop("class", None)

    keys = set(dict_.keys())
    if len(keys ^ {"x1", "x2", "y1", "y2"}) == 0:
        return Line(name=name, **dict_)
    elif len(keys ^ {"x", "y"}) == 0:
        return Point(name=name, **dict_)
    elif len(keys ^ {"x", "y", "width", "height"}) == 0:
        return Rectangle(name=name, **dict_)


class Feature:

    def __init__(self, name, schema):
        self.name = name
        self.schema = schema

BALL = Feature("Ball", Rectangle)
GOAL_POST = Feature("Goal Post", Rectangle)
FIELD_LINE = Feature("Field Line", Line)
CORNER = Feature("Corner", Point)
T_JUNCTION = Feature("T Junction", Point)
X_JUNCTION = Feature("X Junction", Point)
PENALTY_SPOT = Feature("Penalty Spot", Point)
CENTER_CIRCLE = Feature("Center Circle", Point)
NAO = Feature("Nao", Rectangle)
NOT_BALL = Feature("Not Ball", Rectangle)

USED_FEATURES = [BALL, GOAL_POST, FIELD_LINE, CORNER, T_JUNCTION, X_JUNCTION,
                 PENALTY_SPOT, CENTER_CIRCLE, NAO]
