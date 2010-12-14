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
#include "BrowFeatureDetector.h"
#include "HaarCascadeObjectDetector.h"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


BrowFeatureDetector::BrowFeatureDetector( )
{
}

BrowFeatureDetector::~BrowFeatureDetector()
{
}

void 
BrowFeatureDetector::detect(cv::Mat image, Rect face, BrowFeatures &features)
{
  vector<vector<Point> > contours;
  Point left, right, top, bottom;
  vector<Vec4i> hierarchy;

  //rectangle(frameCopy, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height), CV_RGB(255, 0 , 0));
  int w = face.width;
  int h = face.height;
  int scale = 7;                           //determined by testing 
	
  Rect rbrowROI(face.x+cvRound(face.width/2.0), face.y + cvRound(face.height/6.0), cvRound(face.width/2.0)-cvRound(w/scale), cvRound(face.height/5.0));
  Rect lbrowROI(face.x+cvRound(w/scale),  face.y + cvRound(face.height/6.0), cvRound(face.width/2.0)-(w/scale), cvRound(face.height/5.0));
	    
  //rectangle(frameCopy, Point(rbrowROI.x, rbrowROI.y), Point(rbrowROI.x+rbrowROI.width, rbrowROI.y+rbrowROI.height), CV_RGB(0, 255, 255));
  //rectangle(frameCopy, Point(lbrowROI.x, lbrowROI.y), Point(lbrowROI.x+lbrowROI.width, lbrowROI.y+lbrowROI.height), CV_RGB(0, 255, 255));
	    
	
  Mat pyr;      
  Mat edge_detected = image(rbrowROI);
  Mat temp;                
	      
  //pyrDown(edge_detected, pyr, Size(edge_detected.cols/2, edge_detected.rows/2));
  //pyrUp(pyr, edge_detected, edge_detected.size());
	
  //imshow("grayR", edge_detected);                                        
  Canny(edge_detected, temp, 50, 200, 3);  //messing around with temp for different filters and thresholds
  threshold(edge_detected, temp, 125, 255, THRESH_BINARY_INV);
  //imshow("temp", temp);
	
  GaussianBlur(edge_detected, edge_detected, Size(9,9), 2.2, 2.2);
  equalizeHist(edge_detected, edge_detected);
  //imshow("stage1R", edge_detected);
	
	
  threshold(edge_detected, edge_detected, 125, 255, THRESH_BINARY_INV);
  //adaptiveThreshold(edge_detected, edge_detected, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 0);	
  Canny(edge_detected, edge_detected, 50, 200, 3);
  equalizeHist(edge_detected, edge_detected);
  //imshow("browR", edge_detected);
  findContours(edge_detected, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 
  //drawContours(frameCopy, contours, -1, CV_RGB(0, 255, 255 ), 1, CV_AA, hierarchy, 0, Point(rbrowROI.x, rbrowROI.y));
	
  //rbrow = makin points
  int rbrow_contour = 0;
  for(int x=0; x<contours.at(rbrow_contour).size(); x++) {
    Point p = contours.at(rbrow_contour).at(x);
    Point p2 = contours.at(rbrow_contour).at(contours.at(rbrow_contour).size()-x-1);
      
    if(x == 0) {
      left  =  p;
      right =  p;
      top   =  p;
    }
    else {
      if(left.x > p.x) {
        left = p;
      }
        
      if(right.x < p.x || (right.x == p.x && right.y > p.y)) {
        right = p;
      }
            
      if( (abs(top.x - rbrowROI.width/2) > abs(p2.x - rbrowROI.width/2 ))) {
        top = p2;
      }
    }
  }
	
  
  features.right_brow_left   = Point(left.x + rbrowROI.x, left.y + rbrowROI.y);
  features.right_brow_center =  Point(top.x + rbrowROI.x, right.y+rbrowROI.y);
  features.right_brow_right  = Point(right.x + rbrowROI.x, right.y + rbrowROI.y);
  //circle(frameCopy, Point(left.x + rbrowROI.x, left.y + rbrowROI.y), 3, CV_RGB(255,0,0), -1);
  //circle(frameCopy, Point(right.x + rbrowROI.x, right.y + rbrowROI.y), 3, CV_RGB(255,0,0), -1);
  //circle(frameCopy, Point(top.x + rbrowROI.x, right.y+rbrowROI.y), 3, CV_RGB(255,0,0),-1);
  //left brow
        
  edge_detected = image(lbrowROI);
	      
  //pyrDown(edge_detected, pyr, Size(edge_detected.cols/2, edge_detected.rows/2));
  //pyrUp(pyr, edge_detected, edge_detected.size());
  //imshow("grayL", edge_detected);
	
  GaussianBlur(edge_detected, edge_detected, Size(9,9), 2.2, 2.2);
  //imshow("firstaftergaussian", edge_detected);
  equalizeHist(edge_detected, edge_detected);
  //imshow("stageL", edge_detected);
	
  threshold(edge_detected, edge_detected, 150, 255, THRESH_BINARY_INV);
  //imshow("aftergaussian", edge_detected);
  //adaptiveThreshold(edge_detected, edge_detected, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 0);
  Canny(edge_detected, edge_detected, 50, 200, 3);
  equalizeHist(edge_detected, edge_detected);
	
  //adapative
  //imshow("browL", edge_detected);
  findContours(edge_detected, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 
  //drawContours(frameCopy, contours, -1, CV_RGB(0, 255, 255 ), 1, CV_AA, hierarchy, 0, Point(lbrowROI.x, lbrowROI.y));
	
  int lbrow_contour = 0;
  for(int x=0; x<contours.at(lbrow_contour).size(); x++) {
    Point p = contours.at(lbrow_contour).at(x);
    Point p2 = contours.at(lbrow_contour).at(contours.at(lbrow_contour).size()-x-1);
      
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
            
      if( (abs(top.x - lbrowROI.width/2) > abs(p2.x - lbrowROI.width/2 ))) {
        top = p2;
      }
        
      if( (abs(bottom.x - lbrowROI.width/2) > abs(p.x - lbrowROI.width/2 ))) {
        bottom = p;
      }
    }
  }
  //circle(frameCopy, Point(left.x + lbrowROI.x, left.y + lbrowROI.y), 3, CV_RGB(255,0,0), -1);
  //circle(frameCopy, Point(right.x + lbrowROI.x, right.y + lbrowROI.y), 3, CV_RGB(255,0,0), -1);
  //circle(frameCopy, Point(top.x + lbrowROI.x, right.y+lbrowROI.y), 3, CV_RGB(255,0,0),-1);
  features.left_brow_left   =  Point(left.x + lbrowROI.x, left.y + lbrowROI.y);
  features.left_brow_center =  Point(top.x + lbrowROI.x, right.y+lbrowROI.y);
  features.left_brow_right  =  Point(right.x + lbrowROI.x, right.y + lbrowROI.y);

}

