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
#include "MouthFeatureDetector.h"
#include "HaarCascadeObjectDetector.h"

using namespace cv;
using namespace std;


MouthFeatureDetector::MouthFeatureDetector(string cascade_file)
{
  this->detector = new HaarCascadeObjectDetector(cascade_file);
}

MouthFeatureDetector::MouthFeatureDetector(CascadeClassifier cascade)
{
  this->detector = new HaarCascadeObjectDetector(cascade);
}

MouthFeatureDetector::~MouthFeatureDetector()
{
  delete this->detector;
}

void 
MouthFeatureDetector::detect(cv::Mat image, Rect face, MouthFeatures &features)
{
  int face_x = face.x;
  int face_y = face.y;
  int face_w = face.width;
  int face_h = face.height;

  image = image.clone();

  /*Get the region of interest for the mouth*/
  Rect mouthROI(face_x, face_y + cvRound(face_h/2.0), face_w, cvRound(face_h/2.0)); 
  
  /*Detect the lips in the face region*/
  vector<Rect> lips;
  detector->detect(image, lips, mouthROI);

  vector<Rect>::const_iterator lips_it;
 
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  Point left, right, top, bottom;

  for(lips_it = lips.begin(); lips_it != lips.end(); lips_it++) {
    Mat edge_detected = image(*lips_it);
    equalizeHist(edge_detected, edge_detected);

    Mat pyr;
    /*Remove some noise by down/upsampling*/
    pyrDown(edge_detected, pyr, Size(edge_detected.cols/2, edge_detected.rows/2));
    pyrUp(pyr, edge_detected, edge_detected.size());

    /*Remove some additional edge noise*/
    GaussianBlur(edge_detected, edge_detected, Size(9,9), 2.2, 2.2); 
    equalizeHist(edge_detected, edge_detected); /*Equalize color histogram*/

    threshold(edge_detected, edge_detected, 125, 255, THRESH_BINARY); /*Create color blobs*/
    dilate(edge_detected, edge_detected, Mat(), Point(-1,-1),1); /*Fill in gaps*/

    /*Handle edge detection*/
    Canny(edge_detected, edge_detected, 50, 200, 3);
    dilate(edge_detected, edge_detected, Mat(), Point(-1,-1), 2); /*Fill in gaps*/
    erode(edge_detected, edge_detected, Mat(), Point(-1,-1), 1); /*Erode some of the edges*/

    threshold(edge_detected, edge_detected, 200, 255, THRESH_BINARY); /*Create edge blobs*/
          
    Mat tmp = edge_detected;
    findContours(tmp, contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE, cvPoint(0,0)); 
    
    int mouth_contour = 0;
    //assume the contour with the most points is the mouth -- this may be flawed :D
    for(int x=0; x<contours.size(); x++) {
      if(contours.at(x).size() > mouth_contour) {
              mouth_contour = x;
      }
    }

    //drawContours(frameCopy, contours, mouth_contour, CV_RGB(0, 255, 255 ), 1, CV_AA, hierarchy, 0, Point(lips_it->x, lips_it->y));

    /*Find points of interest on the CONTOUR*/
    for(int x=0; x<contours.at(mouth_contour).size(); x++) {
      Point p = contours.at(mouth_contour).at(x);
      Point p2 = contours.at(mouth_contour).at(contours.at(mouth_contour).size()-x-1);
      
      if(x == 0) {
        left  =  p;
        right =  p;
        top   =  p;
        bottom = p;
      }
      else {
        if(left.x > p.x) {
          left = p;
        }
        
        if(right.x < p.x || (right.x == p.x && right.y > p.y)) {
          right = p;
        }
            
        if( (abs(top.x - lips_it->width/2) > abs(p2.x - lips_it->width/2 ))) {
          top = p2;
        }
        
        if( (abs(bottom.x - lips_it->width/2) > abs(p.x - lips_it->width/2 ))) {
          bottom = p;
        }
      }
    }

    //circle(frameCopy, Point(left.x + lips_it->x, left.y + lips_it->y), 3, CV_RGB(255, 0 , 0 ), -1);
    //circle(frameCopy, Point(right.x + lips_it->x, right.y + lips_it->y), 3, CV_RGB(255, 0 , 0 ), -1);
    //circle(frameCopy, Point(top.x + lips_it->x, top.y + lips_it->y), 3, CV_RGB(255, 0 , 0 ), -1);
    //circle(frameCopy, Point(bottom.x + lips_it->x, bottom.y + lips_it->y), 3, CV_RGB(255, 0 , 0 ), -1);       
         
    //right  circle(frameCopy, Point(cvRound(lips_it->x), cvRound(lips_it->y+lips_it->height/2.0)), 3, CV_RGB(255, 0 , 0 ), -1);
    //circle(frameCopy, Point(cvRound(lips_it->x+lips_it->width), cvRound(lips_it->y+lips_it->height/2.0)), 3, CV_RGB(255, 0 , 0 ), -1);
    features.lip_left_edge     = Point(left.x + lips_it->x, left.y + lips_it->y);     /*Point representing the left edge of the lip*/
    features.lip_right_edge    = Point(right.x + lips_it->x, right.y + lips_it->y);    /*Point representing the right edge of the lip*/
    features.lip_top_center    = Point(top.x + lips_it->x, top.y + lips_it->y);      /*Point representing the top center of the lip*/
    features.lip_bottom_center = Point(bottom.x + lips_it->x, bottom.y + lips_it->y);   /*Point representing the bottom center of the lip*/
  }
}

