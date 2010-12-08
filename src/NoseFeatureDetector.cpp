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
#include <iostream>

#include <vector>
#include <opencv2/objdetect/objdetect.hpp>
#include "HaarCascadeObjectDetector.h"
#include "NoseFeatureDetector.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

NoseFeatureDetector::NoseFeatureDetector(string cascade_file)
{
  this->detector = new HaarCascadeObjectDetector(cascade_file);
}

NoseFeatureDetector::NoseFeatureDetector(CascadeClassifier cascade)
{
  this->detector = new HaarCascadeObjectDetector(cascade);
}

NoseFeatureDetector::~NoseFeatureDetector()
{
  delete this->detector;
}

int 
NoseFeatureDetector::detect(Mat image, Rect face, NoseFeatures &nose)
{ 
  Rect noseROI(face.x + cvRound(face.width/4.0), face.y + cvRound(face.height/2.0), cvRound(2.0*face.width/4.0),cvRound(face.height/2.0));
    //cvNamedWindow("result2", 1);
  vector<Rect> noses;
  detector->detect(image, noses, noseROI);

  vector<Rect>::const_iterator nose_it;
  Mat frameCopy = image;
  for(nose_it = noses.begin(); nose_it != noses.end(); nose_it++) {
    //rectangle(frameCopy, Point(nose_it->x, nose_it->y), Point(nose_it->x+nose_it->width, nose_it->y+nose_it->height), CV_RGB(0, 0 , 255 ));
    Point center(nose_it->x + cvRound(nose_it->width/2.0), nose_it->y + cvRound(nose_it->height/2.0));
    //circle(frameCopy, center, 3,  CV_RGB(255, 0 , 0 ), -1);
    //imshow("result2", frameCopy);
    nose.center = center;
    //cout<<nose.center<<endl;
  }

  return noses.size();
}

