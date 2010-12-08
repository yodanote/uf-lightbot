/**
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 */


#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "StaticObjectDetector.h"

using namespace std;
using namespace cv;

StaticObjectDetector::StaticObjectDetector(double x_percent, double y_percent, double width_percent, double height_percent)
{
    this->xp = x_percent;
    this->yp = y_percent;
    this->wp = width_percent;
    this->hp = height_percent;
}

StaticObjectDetector::~StaticObjectDetector() 
{}

int   
StaticObjectDetector::detect(cv::Mat image, std::vector<cv::Rect>& objects, cv::Rect ROI = cv::Rect())
{
    Mat region;

    if(ROI == Rect()) {
      region = image;
    } 
    else {
      region = image(ROI);
    }

    int x = cvRound(this->xp * region.rows());
    int y = cvRound(this->yp * region.cols());
    int w = cvRound(this->wp * region.rows());
    int h = cvRound(this->hp * region.cols());

    /*Make sure its within the region*/
    if(x + w > region.rows()) {
      w = region.rows() - x;
    }
    
    if(y + h > region.cols()) {
      h = regions.cols() - y;
    }
    
    if(ROI != Rect()) {
      x = x + ROI.x;
      y = y + ROI.y;
    } 

    objects.push_back(Rect(x, y, w, h));

    return 1;
}


#endif /*STATICOBJECTDETECTOR_H*/
