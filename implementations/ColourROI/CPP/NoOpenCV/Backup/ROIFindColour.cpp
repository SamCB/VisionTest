/*
Created on Fri Oct 07 10:22:53 2016

@author: Ethan Jones
*/

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>
#include <climits>

#include "ROIFindColour.h"

using namespace std;

typedef vector<vector<vector<float> > > vector3f;
typedef vector<vector<float> > vector2f;
typedef vector<vector<bool> > vector2b;
typedef vector<vector<int> > vector2i;

int** ROIFindColour(float* baseIm, int* shape, int* numRegions)
{
    /*
    Finds a region of interest based on colour blobs. ADD FULL DESCRIPTION.
    */

    clock_t begin_time = clock();
    clock_t fullTime = clock();
    
    // Convert the image to a more usable format.
    vector3f im(shape[0], vector2f(shape[1], vector<float>(shape[2])));
    for(int y = 0; y < shape[0]; y++)
    {
        for(int x = 0; x < shape[1]; x++)
        {
            for(int c = 0; c < shape[2]; c++)
            {
                im[y][x][c] = (float)baseIm[
                                    y*shape[1]*shape[2] + x*shape[2] + c]/256.0;
            }
        }
    }
    
    std::cout << "copy time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    begin_time = clock();
    
    // Estimate green as median of a sample from the bottom of the image.
    // Certainly not very efficient, but it'll do for now.
    vector2f samples(3);
    samples[0] = vector<float>();
    samples[1] = vector<float>();
    samples[2] = vector<float>();
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
    vector<float> green(3);
    for(int c = 0; c < samples.size(); ++c)
    {
        sort(samples[c].begin(), samples[c].end());
        if(samples[c].size()%2 == 0)
        {
            green[c] = (samples[c][samples[c].size()/2-1] + 
                                             samples[c][samples[c].size()/2])/2;
        }
        else
        {
            green[c] = samples[c][samples[c].size()/2];
        }
    }

    std::cout << "green sample time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    begin_time = clock();
    
    // Look for chunks of stuff that isn't green.
    vector2b notGreen(im.size(), vector<bool>(im[0].size()));
    cout << "not green alloc time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    int notGreenCount = 0;
    for(int y = 0; y < im.size(); y++)
    {
        for(int x = 0; x < im[y].size(); x++)
        {
            // This was bugged (no abs) in the original.
            notGreen[y][x] = (abs(im[y][x][0]-green[0]) + 
                             abs(im[y][x][1]-green[1]) +
                                               abs(im[y][x][2]-green[2])) > 0.5;
            if(notGreen[y][x])
                notGreenCount++;
        }
    }
    
    cout << "not green time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    begin_time = clock();
    
    //cout << "Not Green Count: " << notGreenCount << " of " << shape[0]*shape[1]
    //                                                                    << endl;

    // Connected component analysis preparation.
    vector2i groups(im.size(), vector<int>(im[0].size(), 0));
    vector<int> groupCounts(1, 0);
    groupCounts.reserve(1000);
    vector<int> groupLowXs(1, 0);
    groupLowXs.reserve(1000);
    vector<int> groupHighXs(1, 0);
    groupHighXs.reserve(1000);
    vector<int> groupLowYs(1, 0);
    groupLowYs.reserve(1000);
    vector<int> groupHighYs(1, 0);
    groupHighYs.reserve(1000);
    vector<vector<int> > groupLinks(1, vector<int>(1, 0));
    groupLinks.reserve(1000);
    int numGroups = 0;

    // Prevent big groups. Could probably be faster if integrated into CCA.
    for(int y = 50; y < notGreen.size(); y += 50)
    {
        for(int x = 0; x < notGreen[y].size(); x++)
        {
            if(notGreen[y][x])
                notGreenCount--;
            notGreen[y][x] = false;
        }
    }
    for(int y = 0; y < notGreen.size(); y++)
    {
        for(int x = 0; x < notGreen[y].size(); x += 50)
        {
            if(notGreen[y][x])
                notGreenCount--;
            notGreen[y][x] = false;
        }
    }
    
    cout << endl << "Not Green Count: " << notGreenCount << " of " << 
                                                      shape[0]*shape[1] << endl;
    
    cout << "cc prep time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    begin_time = clock();
    float totalAllocationTime = 0;
    clock_t allocTimeStart = 0;
    clock_t allocTimeStart2 = 0;
    float connTime = 0;
    float insertTime = 0;
    float newGroupTime = 0;
    float ifTime = 0;
    int loopCount = 0;
    bool hasNeighbour = false;
    clock_t t3 = 0;
    float timeT3 = 0;
    
    // High so that they will not be smallest group.
    int topNeighbour = INT_MAX; 
    int leftNeighbour = INT_MAX;
    
    // Connected component analysis.
    for(int y = 0; y < notGreen.size(); y++)
    {
        for(int x = 0; x < notGreen[y].size(); x++)
        {
            loopCount++;
            
            // If this is not a green pixel, group it.
            if(notGreen[y][x] == true)
            {
                // Get all neighbours.
                hasNeighbour = false;
                //allocTimeStart2 = clock();
                if(x > 0 && notGreen[y][x-1]){
                    leftNeighbour = groups[y][x-1];
                    hasNeighbour = true;
                }
                else
                    leftNeighbour = INT_MAX;
                if(y > 0 && notGreen[y-1][x]){
                    topNeighbour = groups[y-1][x];
                    hasNeighbour = true;
                }
                else
                    topNeighbour = INT_MAX;
                //ifTime += float(clock()-allocTimeStart2)/CLOCKS_PER_SEC;
                
                // If there are no neighbours create a new label.
                if(!hasNeighbour)
                {
                    numGroups += 1;
                    groups[y][x] = numGroups;
                    //allocTimeStart = clock();
                    groupLowXs.push_back(x);
                    groupHighXs.push_back(x);
                    groupLowYs.push_back(y);
                    groupHighYs.push_back(y);
                    groupCounts.push_back(1);
                    groupLinks.push_back(vector<int>(1, numGroups));
                    groupLinks.back().reserve(100);
                    //totalAllocationTime += float(clock()-allocTimeStart)/CLOCKS_PER_SEC;
                    //newGroupTime += float(clock()-allocTimeStart)/CLOCKS_PER_SEC;
                }   
                // If there is a neighbour build components.
                else
                {
                    
                    //allocTimeStart = clock();
                    if(topNeighbour < leftNeighbour)
                    {
                        // Set the pixel's group.
                        groups[y][x] = topNeighbour;
                        
                        // Add a parent to the left neighbour if needed.
                        if(leftNeighbour != INT_MAX){
                            //allocTimeStart2 = clock();
                            groupLinks[leftNeighbour].push_back(topNeighbour);
                            push_heap(groupLinks[leftNeighbour].begin(), 
                                      groupLinks[leftNeighbour].end(),
                                                                greater<int>());
                            //insertTime += float(clock()-allocTimeStart2)/CLOCKS_PER_SEC;
                        }
                        
                        // Update bounding box.
                        groupCounts[topNeighbour] += 1;
                        if(groupHighXs[topNeighbour] < x)
                            groupHighXs[topNeighbour] = x; 
                        if(groupHighYs[topNeighbour] < y)
                            groupHighYs[topNeighbour] = y;
                    }
                    else
                    {
                        // Set the pixel's group.
                        groups[y][x] = leftNeighbour;
                        
                        // Add a parent to the top neighbour if needed.
                        if(topNeighbour != INT_MAX){
                            //allocTimeStart2 = clock();
                            groupLinks[topNeighbour].push_back(leftNeighbour);
                            push_heap(groupLinks[topNeighbour].begin(), 
                                      groupLinks[topNeighbour].end(), 
                                                                greater<int>());
                            //insertTime += float(clock()-allocTimeStart2)/CLOCKS_PER_SEC;
                        }
                        
                        // Update bounding box.
                        //t3 = clock();
                        groupCounts[leftNeighbour] += 1;
                        if(groupHighXs[leftNeighbour] < x)
                            groupHighXs[leftNeighbour] = x; 
                        if(groupHighYs[leftNeighbour] < y)
                            groupHighYs[leftNeighbour] = y;
                        //timeT3 += float(clock()-t3)/CLOCKS_PER_SEC;
                    }
                    //connTime += float(clock()-allocTimeStart)/CLOCKS_PER_SEC;
                    
                }
            }
        }
    }
    cout << "Loop Count: " << loopCount << endl;
    cout << "cc time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    cout << "Allocation time: " << totalAllocationTime << endl;
    cout << "Time %: " << (totalAllocationTime/(float(clock()-begin_time) /  CLOCKS_PER_SEC))*100 << endl;
    cout << "Connection creation time: " << connTime << endl;
    cout << "Insert time: " << insertTime << endl;
    cout << "New group time: " << newGroupTime << endl;
    cout << "Neighbour find time: " << ifTime << endl;
    cout << "T3 time: " << timeT3 << endl;
    begin_time = clock();
    
    cout << "Number of groups: " << numGroups << endl;
    
    // Don't need a full second pass as we only need bounding boxes. Combining
    // by grabbing pixel location extremes is sufficient. May be a faster way
    // to implement this.

    // Find the parent group for each group.
    bool changed = true;
    while(changed)
    {
        changed = false;
        for(int group=1; group<groupLinks.size(); group++)
        {
            // Tell all linked groups the lowest linked group.
            for(int owner=1; owner<groupLinks[group].size(); owner++)
            {
                changed = true;
                // Insert in sorted order.
                if(groupLinks[group][owner] != group)
                {
                    groupLinks[groupLinks[group][owner]].push_back(
                                                          groupLinks[group][0]);
                    push_heap(groupLinks[groupLinks[group][owner]].begin(), 
                              groupLinks[groupLinks[group][owner]].end(), 
                                                                greater<int>());
                }
            }
            
            // Delete all but the lowest owner.
            groupLinks[group] = vector<int>(1, groupLinks[group][0]);
        }
    }
    
    cout << "parent finding time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    begin_time = clock();
    
    // Apply combinations.
    for(int group=groupLinks.size()-1; group>=0; group--)
    {
        int owner = groupLinks[group][0];
        if(owner != group)
        {
            groupCounts[owner] += groupCounts[group];
            groupCounts[group] = 0;
            if(groupLowXs[group] < groupLowXs[owner])
                groupLowXs[owner] = groupLowXs[group];
            if(groupLowYs[group] < groupLowYs[owner])
                groupLowYs[owner] = groupLowYs[group];
            if(groupHighXs[group] > groupHighXs[owner])
                groupHighXs[owner] = groupHighXs[group];
            if(groupHighYs[group] > groupHighYs[owner])
                groupHighYs[owner] = groupHighYs[group];
        }
    }

    // Merge groups where density remains good.
    changed = true;
    float thresh = 0.8;
    while(changed)
    {
        changed = false;
        
        // Check ever pair of groups that might be merged.
        for(int group1=0; group1<groupLinks.size(); group1++)
        {
            if(groupCounts[group1] > 0)
            {
                for(int group2=group1+1; group2<groupLinks.size(); group2++)
                {
                    if(groupCounts[group2] > 0)
                    {
                        // If both groups have pixels and density after 
                        // combining is good, combine.
                        int xL = min(groupLowXs[group1], groupLowXs[group2]);
                        int xH = max(groupHighXs[group1],groupHighXs[group2]);
                        int yL = min(groupLowYs[group1], groupLowYs[group2]);
                        int yH = max(groupHighYs[group1],groupHighYs[group2]);
                        int size = (xH-xL+1)*(yH-yL+1);
                        if((float)(groupCounts[group1]+groupCounts[group2]) /
                                        (float)(size) > thresh and size < 40000)
                        {
                            changed = true;
                            groupLowXs[group1] = xL;
                            groupHighXs[group1] = xH;
                            groupLowYs[group1] = yL;
                            groupHighYs[group1] = yH;
                            groupCounts[group1] += groupCounts[group2];
                            groupCounts[group2] = 0;
                        }
                    }
                }
            }
        }
        
        // Work from high threshold down to low threshold. Tends to create nicer 
        // BBs.
        if(changed == false and thresh > 0.4)
        {
            changed = true;
            thresh -= 0.1;
        }
    }

    cout << "merge group time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    begin_time = clock();
    
    // Create ROI from every relevant group.
    vector2i roi;
    for(int group=1; group<numGroups+1; group++)
    {
        // Check the group actually has pixels.
        if(groupCounts[group] > 15)
        {
            // Check if the group is low pixel density (probably field lines).
            int width = groupHighXs[group] - groupLowXs[group] + 1;
            int height = groupHighYs[group] - groupLowYs[group] + 1;
            
            // Check pixel density is high enough for a ball.
            if((float)(groupCounts[group])/(float)(width*height) > 0.3)
            {  
                // Balls are round, and we usually get the top half. Make the BB
                // square if we have a big top half.
                float aspect = (float)(width)/(float)(height);
                if(aspect > 1.0 and aspect < 4.0)
                {
                    groupHighYs[group] = min((int)(groupLowYs[group]+width-1), 
                                                            (int)(im.size()-1));
                    height = width;
                }

                // Check the box is the right shape for a ball.    
                if(abs((float)(width)/(float)(height) - 1.0) < 0.1)
                {
                    // Create a region of interest.
                    roi.push_back(vector<int>(4));
                    roi[roi.size()-1][0] = height;
                    roi[roi.size()-1][1] = width;
                    roi[roi.size()-1][2] = groupLowXs[group];
                    roi[roi.size()-1][3] = groupLowYs[group];
                }
            }
        }
    }
    
    // Make a pointer version for return.
    int** data = new int*[roi.size()];
    for(int item=0; item<roi.size(); item++)
        data[item] = roi[item].data();
    
    cout << "ROI found: " << roi.size() << endl;
    
    // Tell python how many ROI were found.
    *numRegions = roi.size();

    cout << "roi creation time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    cout << "C++ total time: " << float( clock () - fullTime ) /  CLOCKS_PER_SEC << endl << endl;
    
    // Return the ROI found.
    return(data);
}


























































