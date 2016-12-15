import numpy as np
import ctypes
import cv2
import time

# Prepare the c++ function.
_ROIFindColour = ctypes.CDLL("./implementations/ColourROI/CPP/NoOpenCV/ROIFindColour.so")
outType = (ctypes.c_int * 4) * 100
_ROIFindColour[1].argtypes = (np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, 
                              flags='CONTIGUOUS'), ctypes.POINTER(ctypes.c_int), 
                                                          ctypes.c_int, outType)
                                                                  
def ROIFindColour(im, slothFormat=True):

    # Grab the c++ function.
    global _ROIFindColour
    
    # Prepare the array.
    cIm = np.ascontiguousarray(im, dtype=np.float32)
    
    # Prepare the parameters.
    cShape = im.ctypes.shape_as(ctypes.c_int)
    ROI = outType()
    numRegions = ctypes.c_int()
    
    # Find the colour ROI.
    #start = time.clock()
    cROI = _ROIFindColour[1](cIm.ctypes.data, cShape, ctypes.byref(numRegions), 
                                                                            ROI)
    #print("Total time: " + str(time.clock()-start))
    
    # Convert output to python format.
    ROI = np.array(ROI)
    numRegions = numRegions.value
    #print(numRegions)
    #print(ROI[0:numRegions])
    
    # If slot format is requested, use it.
    if slothFormat:
        outROI = []
        for region in xrange(numRegions):
            outROI.append(('Ball', {'height': ROI[region][0], 
                                    'width': ROI[region][1], 
                                     'x': ROI[region][2], 'y': ROI[region][3]}))
        return outROI
    else:
        # Return the raw result.
        return ROI, numRegions

#testIm = cv2.imread("../../../../../data/MobileRobotAndBall1/raw_images/10.jpg")
#ROIFindColour(testIm)







































