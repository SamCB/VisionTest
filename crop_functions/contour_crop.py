import cv2

def retrieve_subsections(img):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        # x, y, w, h
        yield cv2.boundingRect(contour)
