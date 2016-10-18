
/*Created on Fri Oct 07 10:22:53 2016
@author: Ethan Jones
*/

#include <boost/python/detail/wrap_python.hpp>
#include <numpy/ndarrayobject.h>
#include <boost/python.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <ctime>

using namespace std;

typedef vector<vector<vector<float> > > vector3f;
typedef vector<vector<float> > vector2f;
typedef vector<vector<bool> > vector2b;
typedef vector<vector<int> > vector2i;

vector<int> ROIFindColour(cv::Mat im)
{
    /*
    Finds a region of interest based on colour blobs. ADD FULL DESCRIPTION.
    */
    //clock_t begin_time = clock();

    //std::cout << "copy time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    //begin_time = clock();

    // Estimate green as median of a sample from the bottom of the image.
    // Certainly not very efficient, but it'll do for now.
    vector2f samples(3);
    samples[0] = vector<float>();
    samples[1] = vector<float>();
    samples[2] = vector<float>();
    for(int y = 300; y < im.size[0]; y += 5)
    {
        for(int x = 0; x < im.size[1]; x += 5)
        {
            cv::Vec3b colour = im.at<cv::Vec3b>(y, x);
            for(int c = 0; c < im.channels(); ++c)
            {
                samples[c].push_back(colour[c]);
            }
        }
    }
    vector<float> green(3);
    for(int c = 0; c < samples.size(); ++c)
    {
        sort(samples[c].begin(), samples[c].end());
        if(samples[c].size()%2 == 0)
        {
            green[c] = (((float)samples[c][samples[c].size()/2-1]) + 
                                    ((float)samples[c][samples[c].size()/2]))/2;
        } 
        else
        {
            green[c] = (float)samples[c][samples[c].size()/2];
        }
    }

    //std::cout << "green sample time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    //begin_time = clock();

    // Look for chunks of stuff that isn't green.
    vector2b notGreen(im.size[0], vector<bool>(im.size[1]));
    for(int y = 0; y < im.size[0]; y++)
    {
        for(int x = 0; x < im.size[1]; x++)
        {
            cv::Vec3b colour = im.at<cv::Vec3b>(y, x);
            notGreen[y][x] = (((float)colour[0])-green[0]) + 
                             (((float)colour[1])-green[1]) +
                                            (((float)colour[1])-green[2]) > 128;
        }
    }

    //std::cout << "not green time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    //begin_time = clock();

    // Connected component analysis preparation.
    vector2i groups(im.size[0], vector<int>(im.size[1], 0));
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

    //std::cout << "cc prep time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    //begin_time = clock();
    
    //float totalAllocationTime = 0;
    //clock_t allocTimeStart = 0;
    //clock_t allocTimeStart2 = 0;
    //float connTime = 0;
    //float insertTime = 0;
    //float newGroupTime = 0;
    //float ifTime = 0;
    //int loopCount = 0;
    //clock_t t3 = 0;
    //float timeT3 = 0;
    
    // High so that they will not be smallest group.
    int topNeighbour = INT_MAX; 
    int leftNeighbour = INT_MAX;
    bool hasNeighbour = false;
    
    // Connected component analysis.
    for(int y = 0; y < notGreen.size(); y++)
    {
        for(int x = 0; x < notGreen[y].size(); x++)
        {
            //loopCount++;
            
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
    //cout << "Loop Count: " << loopCount << endl;
    //cout << "Allocation time: " << totalAllocationTime << endl;
    //cout << "Time %: " << (totalAllocationTime/(float(clock()-begin_time) /  CLOCKS_PER_SEC))*100 << endl;
    //cout << "Connection creation time: " << connTime << endl;
    //cout << "Insert time: " << insertTime << endl;
    //cout << "New group time: " << newGroupTime << endl;
    //cout << "Neighbour find time: " << ifTime << endl;
    //cout << "T3 time: " << timeT3 << endl;
    
    //cout << "cc time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    //begin_time = clock();
    
    //cout << "Number of groups: " << numGroups << endl;
    
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

    //std::cout << "parent finding time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    //begin_time = clock();

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

    //std::cout << "merge group time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    //begin_time = clock();

    // Create ROI from every relevant group.
    //vector2i roi;
    std::vector<int> roi;
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
                                                           (int)(im.size[0]-1));
                    height = width;
                }

                // Check the box is the right shape for a ball.    
                if(abs((float)(width)/(float)(height) - 1.0) < 0.1)
                {
                    // Create a region of interest.
                    // roi.push_back(vector<int>(4));
                    // roi[roi.size()-1][0] = height;
                    // roi[roi.size()-1][1] = width;
                    // roi[roi.size()-1][2] = groupLowXs[group];
                    // roi[roi.size()-1][3] = groupLowYs[group];
                    roi.push_back(height);
                    roi.push_back(width);
                    roi.push_back(groupLowXs[group]);
                    roi.push_back(groupLowYs[group]);
                }
            }
        }
    }
    //std::cout << "roi creation time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    
    // Return the ROI found.
    return(roi);
}

// wrap c++ array as numpy array
static boost::python::object vectorWrapper (cv::Mat baseIm) {
    const clock_t begin_time = clock();
    std::vector<int> const& vec = ROIFindColour(baseIm);
    std::cout << "overall c++ runtime: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    npy_intp size = vec.size();

     // const_cast is rather horrible but we need a writable pointer
     //   in C++11, vec.data() will do the trick
     //   but you will still need to const_cast
     

    int * data = size ? const_cast<int *>(&vec[0]) 
    : static_cast<int *>(NULL); 

    PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_INT, data );
    boost::python::handle<> handle( pyObj );
    boost::python::numeric::array arr( handle );

     // The problem of returning arr is twofold: firstly the user can modify
     //   the data which will betray the const-correctness 
     //   Secondly the lifetime of the data is managed by the C++ API and not the 
     //   lifetime of the numpy array whatsoever. But we have a simple solution..
     

    return arr.copy(); // copy the object. numpy owns the copy now.
}

using namespace boost::python;

BOOST_PYTHON_MODULE(colourROI)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    //conversion requires https://github.com/Algomorph/pyboostcvconverter

    def("ROIFindColour", vectorWrapper);
}
