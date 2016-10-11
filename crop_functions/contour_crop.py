import cv2

def retrieve_subsections(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 200)
    cv2.imshow("edges", edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        # x, y, w, h
        yield cv2.boundingRect(contour)
