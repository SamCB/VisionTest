import numpy as np
import ctypes
import cv2
import time

_ROIFindColour = ctypes.CDLL("ROIFindColour.dll")
_ROIFindColour[1].argtypes = (np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, 
                              flags='CONTIGUOUS'), ctypes.POINTER(ctypes.c_int), 
                                                                   ctypes.c_int)
_ROIFindColour[1].restype = ctypes.POINTER(ctypes.POINTER(
                                                                  ctypes.c_int))
                                                                  
def ROIFindColour(im):
    global _ROIFindColour
    cIm = np.ascontiguousarray(im, dtype=np.float32)
    print(cIm.shape)
    #cIm = cIm.ctypes.data
    print(cIm)
    cShape = im.ctypes.shape_as(ctypes.c_int)
    print(cShape)
    numRegions = ctypes.c_int()
    start = time.clock()
    cROI = _ROIFindColour[1](cIm.ctypes.data, cShape, ctypes.byref(numRegions))
    print(time.clock()-start)
    tempROI = np.ctypeslib.as_array(cROI, (numRegions.value, 4))
    print(tempROI)

testIm = cv2.imread("../../../../data/MobileRobotAndBall1/raw_images/10.jpg")
ROIFindColour(testIm)