/*
Created on Fri Oct 07 10:22:53 2016

@author: Ethan Jones
*/

#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdlib>

using namespace std;

typedef vector<vector<vector<float> > > vector3f;
typedef vector<vector<float> > vector2f;
typedef vector<vector<bool> > vector2b;
typedef vector<vector<int> > vector2i;

vector<int> ROIFindColour(vector<vector<vector<uint8_t> > > baseIm)
{
    /*
    Finds a region of interest based on colour blobs. ADD FULL DESCRIPTION.
    */

    // Convert the image to a more usable format.
    vector3f im = baseIm;
    for(vector3f::iterator y = im.begin(); y != im.end(); ++y)
    {
        for(vector2f::iterator x = im[y].begin(); x != im[y].end(); ++x)
        {
            for(vector<float>::iterator c = im[y][x].begin(); 
                                                       c != im[y][x].end(); ++c)
            {
                im[x][y][c] /= 256.0;
            }
        }
    }
    
    // Estimate green as median of a sample from the bottom of the image.
    // Certainly not very efficient, but it'll do for now.
    vector2f samples = new vector2f(3);
    samples[0] = new vector<float>());
    samples[1] = new vector<float>());
    samples[2] = new vector<float>());
    for(int y = 300; y < im.size(); y += 5)
    {
        for(int x = 0; x < im[y].size(); x += 5)
        {
            for(int c = 0; c < im[y][x].size(); ++c)
            {
                samples[c].push_back(im[y][x][c]);
            }
        }
    }
    vector<float> green = new vector<float>(3);
    for(int c = 0; c < samples.size(); ++c)
    {
        sort(samples[c].begin(), samples[c].end());
        if(samples[c].size()%2 == 0)
        {
            green[c] = (samples[samples[c].size()/2-1][c] + 
                                             samples[samples[c].size()/2][c])/2;
        }
    }

    // Look for chunks of stuff that isn't green.
    vector2b notGreen = new vector2b(im.size());
    for(int y = 0; y < im.size(); y++)
    {
        notGreen[y] = new vector<bool>(im[0].size());
    }
    for(int y = 0; y < im.size(); y++)
    {
        for(int x = 0; x < im[y].size(); x++)
        {
            // This was bugged (no abs) in the original.
            notGreen[y][x] = (abs(im[y][x][0]-green[0]) + 
                             abs(im[y][x][1]-green[1]) +
                                               abs(im[y][x][2]-green[2])) > 0.5;
        }
    }

    // Connected component analysis preparation.
    vector2i groups = new vector2i(im.size());
    for(int y = 0; y < im.size(); y++)
    {
        notGreen[y] = new vector<bool>(im[0].size());
    }
    vector<int> groupsCounts = new vector<int>(1, 0);
    vector<int> groupsLowX = new vector<int>(1, 0);
    vector<int> groupsHighX = new vector<int>(1, 0);
    vector<int> groupsLowY = new vector<int>(1, 0);
    vector<int> groupsHighY = new vector<int>(1, 0);
    vector<vector<int> > groupsLinks = 
                             new vector<vector<int> >(1, new vector<int>(1, 0));
    int numGroups = 0;

    // Prevent big groups. Could probably be faster if integrated into CCA.
    for(int y = 50; y < notGreen.size(); y += 50)
    {
        for(int x = 0; x < notGreen[y].size(); x++)
        {
            notGreen[y][x] = false;
        }
    }
    for(int y = 0; y < notGreen.size(); y++)
    {
        for(int x = 0; x < notGreen[y].size(); x += 50)
        {
            notGreen[y][x] = false;
        }
    }
    
    // Connected component analysis.
    for(int y = 0; y < notGreen.size(); y++)
    {
        for(int x = 0; x < notGreen[y].size(); x++)
        {
            
            # If this is not a green pixel, group it.
            if(notGreen[y][x] == true:
            {
                # Get all neighbours.
                vector<int> neighbours = new vector<int>()
                if(x > 0 && notGreen[y][x-1] == true):
                    neighbours.push_back(groups[y][x-1])
                if(x > 0 && y > 0 && notGreen[y-1][x-1] == true):
                    neighbours.push_back(groups[y-1][x-1])
                if(y > 0 && notGreen[y-1][x] == true):
                    neighbours.push_back(groups[y-1][x])
                if(x < notGreen[0].size()-1 && y > 0 && notGreen[y-1][x+1] == true):
                    neighbours.push_back(groups[y-1][x+1])
                    
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
            }
        }
    }
                            
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




























































