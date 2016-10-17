
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

vector<int> ROIFindColour(cv::Mat baseIm)
{
    /*
    Finds a region of interest based on colour blobs. ADD FULL DESCRIPTION.
    */
    clock_t begin_time = clock();

    // Convert the image to a more usable format.
    vector3f im(baseIm.size[0]);
    for(int y = 0; y < baseIm.size[0]; y++)
    {
        im[y] = vector2f(baseIm.size[1]);
        for(int x = 0; x < baseIm.size[1]; x++)
        {
            im[y][x] = vector<float>(baseIm.channels());
            cv::Vec3b colour = baseIm.at<cv::Vec3b>(y, x);
            for(int c = 0; c < baseIm.channels(); c++)
            {
                im[y][x][c] = (float)(colour.val[c])/256.0;
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
    vector2b notGreen(im.size());
    for(int y = 0; y < im.size(); y++)
    {
        notGreen[y] = vector<bool>(im[0].size());
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

    std::cout << "not green time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    begin_time = clock();

    // Connected component analysis preparation.
    vector2i groups(im.size());
    for(int y = 0; y < im.size(); y++)
    {
        //notGreen[y] = vector<bool>(im[0].size());
        groups[y] = vector<int>(im[y].size(), 0);
    }
    vector<int> groupCounts(1, 0);
    vector<int> groupLowXs(1, 0);
    vector<int> groupHighXs(1, 0);
    vector<int> groupLowYs(1, 0);
    vector<int> groupHighYs(1, 0);
    vector<vector<int> > groupLinks(1, vector<int>(1, 0));
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

    std::cout << "cc prep time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    begin_time = clock();
    
    // Connected component analysis.
    for(int y = 0; y < notGreen.size(); y++)
    {
        for(int x = 0; x < notGreen[y].size(); x++)
        {
            
            // If this is not a green pixel, group it.
            if(notGreen[y][x] == true)
            {
                // Get all neighbours.
                vector<int> neighbours;
                if(x > 0 && notGreen[y][x-1] == true)
                    neighbours.push_back(groups[y][x-1]);
                if(x > 0 && y > 0 && notGreen[y-1][x-1] == true)
                    neighbours.push_back(groups[y-1][x-1]);
                if(y > 0 && notGreen[y-1][x] == true)
                    neighbours.push_back(groups[y-1][x]);
                if(x < notGreen[0].size()-1 && y > 0 && 
                                                     notGreen[y-1][x+1] == true)
                    neighbours.push_back(groups[y-1][x+1]);
                    
                // If there are no neighbours create a new label.
                if(neighbours.size() == 0)
                {
                    numGroups += 1;
                    groups[y][x] = numGroups;
                    groupLowXs.push_back(x);
                    groupHighXs.push_back(x);
                    groupLowYs.push_back(y);
                    groupHighYs.push_back(y);
                    groupCounts.push_back(1);
                    groupLinks.push_back(vector<int>(1, numGroups));
                }   
                // If there is a neighbour build components.
                else
                {
                    groups[y][x] = *min_element(neighbours.begin(), 
                                                              neighbours.end());
                    groupCounts[groups[y][x]] += 1;
                    if(groupHighXs[groups[y][x]] < x)
                        groupHighXs[groups[y][x]] = x; 
                    if(groupHighYs[groups[y][x]] < y)
                        groupHighYs[groups[y][x]] = y;
                    for(int label=0; label<neighbours.size(); label++)
                    {
                        if(neighbours[label] != groups[y][x])
                        {
                            // Insert in sorted order.
                            groupLinks[neighbours[label]].insert(upper_bound(
                                groupLinks[neighbours[label]].begin(),
                                groupLinks[neighbours[label]].end(),
                                                   groups[y][x]), groups[y][x]);
                        }
                    }
                }
            }
        }
    }

    std::cout << "cc time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    begin_time = clock();

    // Don't need a full second pass as we only need bounding boxes. Combining
    // by grabbing pixel location extremes is sufficient. May be a faster way
    // to implement this.

    // Find the parent group for each group.
    bool changed = true;
    while(changed)
    {
        changed = false;
        for(int group=0; group<groupLinks.size(); group++)
        {
            // Tell all linked groups the lowest linked group.
            for(int owner=1; owner<groupLinks[group].size(); owner++)
            {
                changed = true;
                // Insert in sorted order.
                if(groupLinks[group][owner] != group) 
                {
                    groupLinks[groupLinks[group][owner]].insert(upper_bound(
                        groupLinks[groupLinks[group][owner]].begin(),
                        groupLinks[groupLinks[group][owner]].end(),
                                      groupLinks[group][0]), groupLinks[group][0]);
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

    std::cout << "parent finding time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    begin_time = clock();

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
            thresh -= 0.05;
        }
    }

    std::cout << "merge group time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    begin_time = clock();

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
                                                            (int)(im.size()-1));
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
    std::cout << "roi creation time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    
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
