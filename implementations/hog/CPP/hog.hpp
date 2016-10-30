#ifndef HOG_H
#define HOG_H

#include <opencv2/opencv.hpp>
#include <string>

class HogDescriptor {
public:
    HogDescriptor(cv::Mat img, int cellSize, int numBins, double threshold);
    //~HogDescriptor();

    cv::Mat computeHog();
    void visualise();
    cv::Mat visualise_scale(double zoom, double scale);
    void findObjects();
    void filterLines();

private:
    int cellSize{8};
    int numBins{9};
    double threshold{15};
    cv::Mat img;
    cv::Mat gradients;
    cv::Mat mask;

    void drawLines(cv::Mat& output, double zoom, double scale);
};

#endif