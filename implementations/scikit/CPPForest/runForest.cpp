#include "cppForest.hpp"
#include <vector>

int main()
{
    int i = 0;

    for(i; i<10000000; i++)
        classify(std::vector<float>(16, 1.0/16.0));
    
    return 0;
}

/*
std::vector<float> makeHistogram(cv::Mat image)
{

}*/
