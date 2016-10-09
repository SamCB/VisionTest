# encoding=utf8

def subarea_crop(crops):
    """Take an iterator of crops we will make and add aditional subareas

    Example:

                 Nao: 15, 224, 50, 192
                 ╔════════════════════╗
             ^   ║        ----        ║
             |   ║      ⎛      ⎞      ║
             |   ║      ⎝      ⎠      ║
             |   ║    +--+----+--+    ║
             |   ║ +--+          +--+ ║
             |   ║ |  |          |  | ║
       580mm |   ║ +--+          +--+ ║
             |   ║    |          |  We assume any bounding box is
             |   ║    +---+--+---+  a Nao and there is a ball at
             |   ║    |   |  |   |  it's foot.
             |   ║    |   |  |   |  Create extra bounding boxes in
          ^  |   ║ +---+  |  |   |  the bottom 6th of what we've
    100mm |  v   ║ |   |--+  +---+  been given.
          |      ║ |   |              ║
          v      ║ +---+              ║
                 ╚════════════════════╝
                   <--->
                   100mm

    So this example will yield:
        (15, 244, 50, 192),
        (15, 352, 32, 32),
        (31, 352, 32, 32),
        (47, 352, 32, 32)
        (63, 352, 32, 32)
    """
    for x, y, w, h in crops:
        yield x, y, w, h

        # Ignore landscape boxes or boxes where the subarea will be too small
        if w > h or h < 100:
            continue

        sub_x = x
        sub_h = h / 6.
        sub_y = int(y + sub_h * 5)
        sub_h = int(sub_h)
        while sub_x < x + w:
            yield sub_x, sub_y, sub_h, sub_h
            # Shuffle bounding box half a boxwidth to the right
            # +1 in case the subbox is only 1 pixel wide
            sub_x += sub_h//2 + 1
