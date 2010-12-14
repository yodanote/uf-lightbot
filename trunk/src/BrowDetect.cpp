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

#include "HaarCascadeObjectDetector.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


int
main(int argc, char *argv[])
{
  CvCapture* capture = NULL;

  double t = 0.0;
  Mat frame, frameCopy;

  Mat gray;
  string fname;

  unsigned int kmod = 0;

  ObjectDetector *faceDetector = new HaarCascadeObjectDetector("haarcascade_frontalface_alt.xml");

  capture = cvCaptureFromCAM(0);

  if( capture ) {
    cout << "In capture ..." << endl;
    vector<Rect> faces;
  for(;;) {
    kmod++;
    IplImage* iplImg = cvQueryFrame( capture );
    frame = iplImg;

    if( frame.empty() )
        break;
    
    if( iplImg->origin == IPL_ORIGIN_TL )
      frame.copyTo( frameCopy );
    else 
      flip( frame, frameCopy, 0 );

      cvtColor(frameCopy, gray, CV_BGR2GRAY);
    if(kmod%2 == 0) {
      namedWindow("result", 1);
        faces.clear();
      t = (double)cvGetTickCount();
      faceDetector->detect(gray, faces);
      t = (double)cvGetTickCount() - t;
     //cout<<"execution time = "<<(t/((double)cvGetTickFrequency()*1000.0))<<" ms"<<endl;
	
    }
      imshow("result", frameCopy);
      vector<Rect>::const_iterator it;
      vector<vector<Point> > contours;
     Point left, right, top, bottom;
      vector<Vec4i> hierarchy;
      namedWindow("brow", 1);
      for(it = faces.begin(); it != faces.end(); it++) {
        rectangle(frameCopy, Point(it->x, it->y), Point(it->x+it->width, it->y+it->height), CV_RGB(255, 0 , 0));
	int w = it->width;
	int h = it->height;
	int scale = 7;                           //determined by testing 
	Rect rbrowROI(it->x+cvRound(it->width/2.0), it->y + cvRound(it->height/6.0), cvRound(it->width/2.0)-cvRound(w/scale), cvRound(it->height/5.0));
        Rect lbrowROI(it->x+cvRound(w/scale),  it->y + cvRound(it->height/6.0), cvRound(it->width/2.0)-(w/scale), cvRound(it->height/5.0));
	    
	rectangle(frameCopy, Point(rbrowROI.x, rbrowROI.y), Point(rbrowROI.x+rbrowROI.width, rbrowROI.y+rbrowROI.height), CV_RGB(0, 255, 255));
	rectangle(frameCopy, Point(lbrowROI.x, lbrowROI.y), Point(lbrowROI.x+lbrowROI.width, lbrowROI.y+lbrowROI.height), CV_RGB(0, 255, 255));
	    
	
	      //points
	      
	
	
	Mat pyr;
	      
	//right brow
	      
	Mat edge_detected = gray(rbrowROI);
	Mat temp;                
	      
	//pyrDown(edge_detected, pyr, Size(edge_detected.cols/2, edge_detected.rows/2));
	//pyrUp(pyr, edge_detected, edge_detected.size());
	imshow("grayR", edge_detected);                                        
	Canny(edge_detected, temp, 50, 200, 3);                   //messing around with temp for different filters and thresholds
	threshold(edge_detected, temp, 125, 255, THRESH_BINARY_INV);
	imshow("temp", temp);
	
	GaussianBlur(edge_detected, edge_detected, Size(9,9), 2.2, 2.2);
	equalizeHist(edge_detected, edge_detected);
	imshow("stage1R", edge_detected);
	
	
        threshold(edge_detected, edge_detected, 125, 255, THRESH_BINARY_INV);
	//adaptiveThreshold(edge_detected, edge_detected, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 0);	
	Canny(edge_detected, edge_detected, 50, 200, 3);
	equalizeHist(edge_detected, edge_detected);
	imshow("browR", edge_detected);
	findContours(edge_detected, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 
	drawContours(frameCopy, contours, -1, CV_RGB(0, 255, 255 ), 1, CV_AA, hierarchy, 0, Point(rbrowROI.x, rbrowROI.y));
	
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
	circle(frameCopy, Point(left.x + rbrowROI.x, left.y + rbrowROI.y), 3, CV_RGB(255,0,0), -1);
        circle(frameCopy, Point(right.x + rbrowROI.x, right.y + rbrowROI.y), 3, CV_RGB(255,0,0), -1);
        circle(frameCopy, Point(top.x + rbrowROI.x, right.y+rbrowROI.y), 3, CV_RGB(255,0,0),-1);
	
	//left brow
        
	edge_detected = gray(lbrowROI);
	      
	//pyrDown(edge_detected, pyr, Size(edge_detected.cols/2, edge_detected.rows/2));
	//pyrUp(pyr, edge_detected, edge_detected.size());
	imshow("grayL", edge_detected);
	
	GaussianBlur(edge_detected, edge_detected, Size(9,9), 2.2, 2.2);
	imshow("firstaftergaussian", edge_detected);
        equalizeHist(edge_detected, edge_detected);
	imshow("stageL", edge_detected);
	
	
        threshold(edge_detected, edge_detected, 150, 255, THRESH_BINARY_INV);
        imshow("aftergaussian", edge_detected);
	//adaptiveThreshold(edge_detected, edge_detected, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 0);
	Canny(edge_detected, edge_detected, 50, 200, 3);
	equalizeHist(edge_detected, edge_detected);
	//adapative
	imshow("browL", edge_detected);
	findContours(edge_detected, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 
	drawContours(frameCopy, contours, -1, CV_RGB(0, 255, 255 ), 1, CV_AA, hierarchy, 0, Point(lbrowROI.x, lbrowROI.y));
	
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
        circle(frameCopy, Point(left.x + lbrowROI.x, left.y + lbrowROI.y), 3, CV_RGB(255,0,0), -1);
        circle(frameCopy, Point(right.x + lbrowROI.x, right.y + lbrowROI.y), 3, CV_RGB(255,0,0), -1);
        circle(frameCopy, Point(top.x + lbrowROI.x, right.y+lbrowROI.y), 3, CV_RGB(255,0,0),-1);
     }
       imshow("result", frameCopy);
      
     

     if( waitKey( 10 ) >= 0 )
         goto _cleanup_;
    }
    }
   _cleanup_:
  cvReleaseCapture( &capture );
  delete faceDetector;
  return 0;
}


	    
         
