import utils
import random


def use():
    return RandomFunction().answer


def random_rect(world_height, world_width, x=None, y=None, width=None, height=None):
    if x is None:
        return {
            'x': random.randrange(0, world_width),
            'y': random.randrange(0, world_height),
            'width': random.randrange(10, 100),
            'height': random.randrange(10, 100)
        }
    else:
        return {
            'x': random.randrange(max(0, x - 10), min(world_width, x + 10) + 1),
            'y': random.randrange(max(0, y - 10), min(world_height, y + 10) + 1),
            'width': random.randrange(width - 2, width + 2 + 1),
            'height': random.randrange(height - 2, height + 2 + 1)
        }


def random_line(world_height, world_width, x1=None, y1=None, x2=None, y2=None):
    if x1 is None:
        return {
            'x1': random.randrange(0, world_width),
            'y1': random.randrange(0, world_height),
            'x2': random.randrange(0, world_width),
            'y2': random.randrange(0, world_height)
        }
    else:
        return {
            'x1': random.randrange(max(0, x1 - 2),
                                   min(world_width, x1 + 2) + 1),
            'y1': random.randrange(max(0, y1 - 2),
                                   min(world_height, y1 + 2) + 1),
            'x2': random.randrange(max(0, x2 - 2),
                                   min(world_width, x2 + 2) + 1),
            'y2': random.randrange(max(0, y2 - 2),
                                   min(world_height, y2 + 2) + 1),
        }


def random_point(world_height, world_width, x=None, y=None):
    if x is None:
        return {
            'x': random.randrange(0, world_width),
            'y': random.randrange(0, world_height)
        }
    else:
        return {
            'x': random.randrange(max(0, x - 2), min(world_width, x + 2) + 1),
            'y': random.randrange(max(0, y - 2), min(world_height, y + 2) + 1)
        }


class RandomFunction:

    def __init__(self):
        self.features = []
        for feature in utils.USED_FEATURES:
            f = {"schema": feature.schema, "name": feature.name}
            if feature.schema == utils.Rectangle:
                f['func'] = random_rect
            elif feature.schema == utils.Line:
                f['func'] = random_line
            elif feature.schema == utils.Point:
                f['func'] = random_point
            self.features.append(f)

    def answer(self, frame):
        output = []
        for f in self.features:
            if 'points' not in f:
                # Initialise points if they don't exist yet
                f['points'] = f['func'](*frame.shape[:2])

            f['points'] = f['func'](*frame.shape[:2], **f['points'])
            output.append((f['name'], f['points']))

        return output
