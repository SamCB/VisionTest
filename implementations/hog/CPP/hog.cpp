#include <boost/python.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include "hog.hpp"


HogDescriptor::HogDescriptor(cv::Mat i, int c, int b, double t): img{i}, cellSize{c}, numBins{b}, threshold{t} {
    int numCellsx = img.size[1]/cellSize;
    int numCellsy = img.size[0]/cellSize;
    int size[3] = {numCellsy, numCellsx, numBins};
    gradients = cv::Mat::zeros(3, size, CV_32F);
    mask = cv::Mat::zeros(2, size, CV_8UC1);
};

cv::Mat HogDescriptor::computeHog() {
    int numCellsx = img.size[1]/cellSize;
    int numCellsy = img.size[0]/cellSize;

    cv::Mat grey;
    cvtColor(img, grey, CV_BGR2GRAY);

    double radPerBin = CV_PI/numBins;
    for (int y = 1; y < grey.rows-1; ++y) {
        for(int x = 1; x < grey.cols-1; ++x) {
            int cellx = x/cellSize;
            int celly = y/cellSize;
            double xdiff = grey.at<char>(y, x+1) - grey.at<char>(y, x-1);
            double ydiff = grey.at<char>(y+1, x) - grey.at<char>(y-1, x);
            double mag = sqrt(xdiff*xdiff + ydiff*ydiff);
            double angle = atan2(ydiff, xdiff);
            if (angle < 0) {
                angle += CV_PI;
            }
            
            if (mag > threshold && cellx < numCellsx && celly < numCellsy) {
                int bin = angle/radPerBin;
                int prevBin = (bin == 0 ? 8 : bin-1);
                int ratio = (angle - bin*radPerBin)/radPerBin;
                gradients.at<float>(celly, cellx, bin) += mag * (1-ratio);
                gradients.at<float>(celly, cellx, prevBin) += mag * ratio;
            }
/*            if (mag > threshold && cellx < numCellsx && celly < numCellsy) {
                int bin = angle/(radPerBin/2);
                bin = ((bin+1)/2)%numBins;
                int prevBin = (bin == 0 ? 8 : bin-1);
                int ratio = (angle - bin*radPerBin + radPerBin/2)/radPerBin;
                gradients.at<double>(celly, cellx, bin) += mag * ratio;
                gradients.at<double>(celly, cellx, prevBin) += mag * (1-ratio);
            }*/
        }
    }

    //normalise cell gradients
    for(int celly = 0; celly < numCellsy; ++celly) {
        for(int cellx = 0; cellx < numCellsx; ++cellx) {
            double sum = 0.0;
            for(int b = 0; b < numBins; ++b) {
                sum += gradients.at<float>(celly, cellx, b);
            }
            if(sum != 0.0) {
                for(int b = 0; b < numBins; ++b) {
                    gradients.at<float>(celly, cellx, b) /= sum;

                }
                mask.at<unsigned char>(celly, cellx) = 1;
            }
        }
    }

    return gradients;

    /*int morph_size = 2;
    cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
    std::cout<<element<<std::endl;
    cv::morphologyEx( mask, mask, cv::MORPH_OPEN, element);
    //cv::dilate( mask, mask, element);*/
}


void HogDescriptor::visualise() {
    drawLines(img, 1, 1);
    //return img;

    /*cv::namedWindow("hog visualization", cv::WINDOW_NORMAL);
    cv::imshow("hog visualization", output);
    cv::namedWindow("mask visualization", cv::WINDOW_NORMAL);
    cv::imshow("mask visualization", mask);
    cv::waitKey(0);*/
    //return output;
}

cv::Mat HogDescriptor::visualise_scale(double zoom, double scale) {
    cv::Mat output;
    resize(img, output, cv::Size( (int)(img.cols*zoom), (int)(img.rows*zoom) ) );
 
    drawLines(output, zoom, scale);

    /*cv::namedWindow("hog visualization", cv::WINDOW_NORMAL);
    cv::imshow("hog visualization", output);
    cv::namedWindow("mask visualization", cv::WINDOW_NORMAL);
    cv::imshow("mask visualization", mask);
    cv::waitKey(0);*/
    return output;
}

void HogDescriptor::drawLines(cv::Mat& output, double zoom, double scale) {
    double radPerBin = (CV_PI/(double)numBins); // dividing 180 into 9 bins, how large (in rad) is one bin?

    int numCellsx = img.size[1]/cellSize;
    int numCellsy = img.size[0]/cellSize;
 
    // draw cells
    for (int celly=0; celly<numCellsy; celly++)
    {
        for (int cellx=0; cellx<numCellsx; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;
 
            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;
 
            cv::rectangle(output, cv::Point((int)(drawX*zoom), (int)(drawY*zoom)), cv::Point((int)((drawX+cellSize)*zoom), (int)((drawY+cellSize)*zoom)), cv::Scalar(100,100,100), 1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<numBins; bin++)
            {
                double currentGradStrength = gradients.at<float>(celly, cellx, bin);
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                double currRad = bin * radPerBin + radPerBin/2;
 
                double dirVecX = cos( currRad );
                double dirVecY = sin( currRad );
                double maxVecLen = (double)(cellSize/2.f);
 
                // compute line coordinates
                double x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                double y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                double x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                double y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visualization
                cv::line(output, cv::Point((int)(x1*zoom),(int)(y1*zoom)), cv::Point((int)(x2*zoom),(int)(y2*zoom)), cv::Scalar(0,0,255), 1);
 
            } // for (all bins)
        } // for (cellx)
    } // for (celly)
}

void HogDescriptor::findObjects() {
    // Setup SimpleBlobDetector parameters.
    cv::SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 255;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 5;
    params.maxArea = 5000;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.01;
    params.maxCircularity = 1;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.01;
    params.maxConvexity = 1;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    params.maxInertiaRatio = 1;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params); 

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(mask, keypoints );


    std::cout << "keypoints:" << std::endl;
    for(auto k : keypoints) {
        std::cout << k.pt << " " << k.size << std::endl;
        //cv::rectangle(img, cv::Point((int)((k.pt.x-k.size)*cellSize), (int)((k.pt.y-k.size)*cellSize)), cv::Point((int)((k.pt.x+k.size)*cellSize), (int)((k.pt.y+k.size)*cellSize)), cv::Scalar(0,0,255), 1);
        cv::circle(img, k.pt*cellSize, k.size*cellSize, cv::Scalar(0,0,255));
    }
     
    // // Show blobs
    cv::imshow("keypoints", img);
    cv::waitKey(0);
}

void HogDescriptor::filterLines() {
    /*for(int y = 1; y < gradients.size[0]-1; ++y) {
        for(int x = 1; x < gradients.size[1]-1; ++x) {
            float diff = 0;

        }
    }*/
}

/*PyObject* pyopencv_from(const cv::Mat& m)
{
  if( !m.data )
      Py_RETURN_NONE;
  cv::Mat temp, *p = (cv::Mat*)&m;
  if(p->allocator != &g_numpyAllocator)
  {
      temp.allocator = &g_numpyAllocator;
      m.copyTo(temp);
      p = &temp;
  }
  p->addref();
  return pyObjectFromRefcount(p->refcount);
}

boost::python::object toPython( const cv::Mat &frame )
{
    PyObject* pyObjFrame = pyopencv_from( frame );
    boost::python::object boostPyObjFrame(boost::python::handle<>((PyObject*)pyObjFrame));

    return boostPyObjFrame;
}*/

using namespace boost::python;

BOOST_PYTHON_MODULE(hog)
{

    //conversion requires https://github.com/Algomorph/pyboostcvconverter

    class_<HogDescriptor>("HogDescriptor", init<cv::Mat, int, int, float>())
        .def("computeHog", &HogDescriptor::computeHog)
        .def("visualise", &HogDescriptor::visualise)
        .def("visualise_scale", &HogDescriptor::visualise_scale)
    ;
}
