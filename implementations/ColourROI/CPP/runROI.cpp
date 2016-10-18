#include "ROIFindColour.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s filename\n", argv[0]);
        return 1;
    }
    cv::Mat img = cv::imread(argv[1]);
    std::vector<int> rois = ROIFindColour(img);
    std::cout << "size: " << rois.size() << std::endl;
    for(int i = 0; i < rois.size()/4; ++i) {
        printf("%d %d %d %d\n", rois[i], rois[i+1], rois[i+2], rois[i+3]);
    }
    return 0;
}