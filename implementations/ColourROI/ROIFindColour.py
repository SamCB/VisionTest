# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 10:22:53 2016

@author: Ethan Jones
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
from skimage.feature import blob_doh
import bisect
import time

def ROIFindColour(im):
    """
    Finds a region of interest based on colour blobs. ADD FULL DESCRIPTION.
    """
    start = time.clock()
    # Convert the image to a more usable format. A Nao version should be able to
    # avoid this.
    im = im.astype(np.float)
    im /= 256

    greenStart = time.clock()
    
    # Estimate green.
    green = np.median(im[300:im.shape[0]:5,0:im.shape[1]:5,:], (0,1))

    # Look for chunks of stuff that isn't green.
    notGreen = np.sum(im - green, 2) > 0.5

    # Slow in python, but would be fast on Nao in C++.
    groups = np.zeros(notGreen.shape, np.int)
    groupsCounts = [0]
    groupsLowX = [0]
    groupsHighX = [0]
    groupsLowY = [0]
    groupsHighY = [0]
    groupsLinks = [[0]]
    numGroups = 0

    # Prevent big groups. Could probably be faster if integrated into CCA.
    notGreen[0:notGreen.shape[0]:50] = 0
    notGreen[:,0:notGreen.shape[1]:50] = 0
    
    print("Green classification took: " + str(time.clock() - greenStart))
    ccaStart = time.clock()
    
    # Connected component analysis (CCA).
    for y in xrange(notGreen.shape[0]):
        for x in xrange(notGreen.shape[1]):
            
            # If this is not a green pixel, group it.
            if notGreen[y, x] == 1:
                
                # Get all neighbours.
                neighbours = []
                if x > 0 and notGreen[y, x-1] == 1:
                    neighbours.append(groups[y, x-1])
                if x > 0 and y > 0 and notGreen[y-1, x-1] == 1:
                    neighbours.append(groups[y-1, x-1])
                if y > 0 and notGreen[y-1, x] == 1:
                    neighbours.append(groups[y-1, x])
                if x < notGreen.shape[1]-1 and y > 0 and \
                                                        notGreen[y-1, x+1] == 1:
                    neighbours.append(groups[y-1, x+1])
                    
                # If there are no neighbours create a new label.
                if len(neighbours) == 0:
                    numGroups += 1
                    groups[y, x] = numGroups
                    groupsLowX.append(x)
                    groupsHighX.append(x)
                    groupsLowY.append(y)
                    groupsHighY.append(y)
                    groupsCounts.append(1)
                    groupsLinks.append([numGroups])
                    
                # If there is a neighbour build components.
                else:
                    groups[y, x] = min(neighbours)
                    groupsCounts[groups[y, x]] += 1
                    if groupsHighX[groups[y, x]] < x:
                        groupsHighX[groups[y, x]] = x  
                    if groupsHighY[groups[y, x]] < y:
                        groupsHighY[groups[y, x]] = y
                    for label in neighbours:
                        if label != groups[y, x]:
                            groupsLinks[label].append(groups[y, x])
                            
    print("CCA took: " + str(time.clock() - ccaStart))
    
    # Sort the group links.
    for linkList in range(len(groupsLinks)):
        groupsLinks[linkList] = np.sort(groupsLinks[linkList]).tolist()

    # Don't need a full second pass as we only need bounding boxes. Combining
    # by grabbing pixel location extremes is sufficient.

    # Find the parent group for each group.
    changed = True
    while changed:
        changed = False
        for group in xrange(len(groupsLinks)):
            
            # Tell all linked groups the lowest linked group.
            for owner in xrange(1, len(groupsLinks[group])):
                changed = True
                bisect.insort_left(groupsLinks[groupsLinks[group][owner]], 
                                                          groupsLinks[group][0])
            
            # Delete all but the lowest owner.
            groupsLinks[group] = [groupsLinks[group][0]]
            
    # Apply combinations.
    for group in xrange(len(groupsLinks)-1, -1, -1):
        owner = groupsLinks[group][0]
        if owner != group:
            groupsCounts[owner] += groupsCounts[group]
            groupsCounts[group] = 0
            if groupsLowX[group] < groupsLowX[owner]:
                groupsLowX[owner] = groupsLowX[group]
            if groupsLowY[group] < groupsLowY[owner]:
                groupsLowY[owner] = groupsLowY[group]
            if groupsHighX[group] > groupsHighX[owner]:
                groupsHighX[owner] = groupsHighX[group]
            if groupsHighY[group] > groupsHighY[owner]:
                groupsHighY[owner] = groupsHighY[group]

    # Merge groups where density remains good.
    changed = True
    thresh = 0.8
    while changed:
        changed = False
        
        # Check ever pair of groups that might be merged.
        for group1 in xrange(len(groupsLinks)):
            if groupsCounts[group1] > 0:
                for group2 in xrange(group1+1, len(groupsLinks)):
                    if groupsCounts[group2] > 0:
                        
                        # If both groups have pixels and density after combining
                        # is good, combine.
                        xL = min([groupsLowX[group1], groupsLowX[group2]])
                        xH = max([groupsHighX[group1], groupsHighX[group2]])
                        yL = min([groupsLowY[group1], groupsLowY[group2]])
                        yH = max([groupsHighY[group1], groupsHighY[group2]])
                        size = (xH-xL+1)*(yH-yL+1)
                        if float(groupsCounts[group1] + groupsCounts[group2])/ \
                                          float(size) > thresh and size < 40000:
                            changed = True
                            groupsLowX[group1] = xL
                            groupsHighX[group1] = xH
                            groupsLowY[group1] = yL
                            groupsHighY[group1] = yH
                            groupsCounts[group1] += groupsCounts[group2]
                            groupsCounts[group2] = 0
        
        # Work from high threshold down to low threshold. Tends to create nicer 
        # BBs.
        if changed == False and thresh > 0.4:
            changed = True
            thresh -= 0.05

    # Create ROI from every relevant group.
    roi = []
    for group in range(1, numGroups+1):
        
        # Check the group actually has pixels.
        if groupsCounts[group] > 15:
            
            # Check if the group is low pixel density (probably field lines).
            width = groupsHighX[group] - groupsLowX[group] + 1
            height = groupsHighY[group] - groupsLowY[group] + 1
            
            # Check pixel density is high enough for a ball.
            if float(groupsCounts[group])/float(width*height) > 0.3:
                
                # Balls are round, and we usually get the top half. Make the BB
                # square if we have a big top half.
                aspect = float(width)/float(height)
                if aspect > 1.0 and aspect < 4.0:
                    groupsHighY[group] = min([groupsLowY[group]+width-1, 
                                                                 im.shape[0]-1])
                    height = width

                # Check the box is the right shape for a ball.    
                if abs(float(width)/float(height) - 1.0) < 0.1:
                
                    # Create a region of interest.
                    x = groupsLowX[group]
                    y = groupsLowY[group]
                    roi.append(('ball', {'height': height, 'width': width, 
                                                               'x': x, 'y': y}))
                                                               
    print("ROI finding took: " + str(time.clock()-start))
    
    # Return the ROI found.
    return roi




























































