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
  ObjectDetector *righteyeDetector = new HaarCascadeObjectDetector("haarcascade_mcs_righteye.xml");
  ObjectDetector *lefteyeDetector = new HaarCascadeObjectDetector("haarcascade_mcs_lefteye.xml");
  ((HaarCascadeObjectDetector*)righteyeDetector)->setImageScale(.7);
  ((HaarCascadeObjectDetector*)lefteyeDetector) ->setImageScale(.7);

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
      cvNamedWindow("result", 1);
        faces.clear();
      t = (double)cvGetTickCount();
      faceDetector->detect(gray, faces);
      t = (double)cvGetTickCount() - t;
     cout<<"execution time = "<<(t/((double)cvGetTickFrequency()*1000.0))<<" ms"<<endl;
    }

      vector<Rect>::const_iterator it;
      vector<Rect>::const_iterator reye_it;
      vector<Rect>::const_iterator leye_it;
      for(it = faces.begin(); it != faces.end(); it++) {
        rectangle(frameCopy, Point(it->x, it->y), Point(it->x+it->width, it->y+it->height), CV_RGB(255, 0 , 0));
     
        //Rect reyeROI((it->x)+(it->width/2.0), it->y + cvRound(it->height/4.0)+10,cvRound(it->width/2.0)-35, cvRound(it->height/4.0)-20);        //for set region of eye (rather than detecting eye)
	//Rect leyeROI(it->x+35, it->y+cvRound(it->height/4.0)+10,cvRound(it->width/2.0-35), cvRound(it->height/4.0)-20);                                 //same as above
	
	float scale = 7.0;
	//Rect reyeROI((it->x)+cvRound(it->width/2.0), it->y + cvRound(it->height/4.0),cvRound(it->width/2.0)-cvRound(it->width/scale), cvRound(it->height/4.0)-cvRound(it->height/(1.3*scale)));       //prob should change constants to factors of face
	//Rect leyeROI(it->x+cvRound(it->x/scale), it->y+cvRound(it->height/4.0), cvRound(it->width/2.0)-cvRound(it->width/scale),cvRound(it->height/4.0)-cvRound(it->height/(1.3*scale)));       
        Rect reyeROI(it->x+cvRound(it->width/2.0), it->y, cvRound(it->width/2.0), cvRound(it->height/2.0));
	Rect leyeROI(it->x, it->y, cvRound(it->width/2.0), cvRound(it->height/2.0));
     
        rectangle(frameCopy, Point(reyeROI.x, reyeROI.y), Point(reyeROI.x + reyeROI.width, reyeROI.y + reyeROI.height), CV_RGB(0, 255, 0));
	rectangle(frameCopy, Point(leyeROI.x, leyeROI.y), Point(leyeROI.x + leyeROI.width, leyeROI.y + leyeROI.height), CV_RGB(0, 255, 0));
	
	
        
       vector<Rect> reye;
       vector<Rect> leye;
       righteyeDetector->detect(gray, reye, reyeROI);
       lefteyeDetector->detect(gray, leye, leyeROI);
	      
	vector<Rect>::const_iterator reye_it;
        vector<Rect>::const_iterator leye_it;
        vector<vector<Point> > contours;
	Point top, bottom;
        vector<Vec4i> hierarchy;
	
	//reye.push_back(Rect((it->x)+(it->width/2.0), it->y + cvRound(it->height/4.0)+10,cvRound(it->width/2.0)-35, cvRound(it->height/4.0)-20));
	//leye.push_back(Rect(it->x+35, it->y+cvRound(it->height/4.0)+10,cvRound(it->width/2.0-35), cvRound(it->height/4.0)-20));
	cvNamedWindow("eyes", 1);
        
       for(reye_it = reye.begin(); reye_it != reye.end(); reye_it++) {
       rectangle(frameCopy, Point(reye_it->x, reye_it->y), Point(reye_it->x+reye_it->width, reye_it->y+reye_it->height), CV_RGB(0, 0 , 255 ));
       
       Mat edge_detected = gray(*reye_it);
		
       Mat pyr, temp;
		
       GaussianBlur(edge_detected, edge_detected, Size(9,9), 2.2, 2.2);
       equalizeHist(edge_detected, edge_detected);
	
       threshold(edge_detected, edge_detected, 125, 255, THRESH_BINARY);
       dilate(edge_detected, edge_detected, Mat(), Point(-1,-1),1);
       Canny(edge_detected, edge_detected, 50, 200, 3);
	
       threshold(edge_detected, edge_detected, 200, 255, THRESH_BINARY);

       dilate(edge_detected, edge_detected, Mat(), Point(-1,-1),1); 
       imshow("eyes", edge_detected);
       findContours(edge_detected, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 
       drawContours(frameCopy, contours, -1, CV_RGB(0,255,255), 1, CV_AA, hierarchy, 0, Point(reye_it->x, reye_it->y));
       imshow("contour", edge_detected);
	
	int reye_contour = 0;
	
        for(int x=0; x<contours.size(); x++) {
            if(contours.at(x).size() > reye_contour) {
              reye_contour = x;
             }
         }
        for(int x=0; x<contours.at(reye_contour).size(); x++) {
        Point p = contours.at(reye_contour).at(x);
        Point p2 = contours.at(reye_contour).at(contours.at(reye_contour).size()-x-1);
      
        if(x == 0) {
         top   =  p;
         bottom = p;
        }
        else {
         if( (abs(top.x - reye_it->width/2) > abs(p2.x - reye_it->width/2 ))) {
           top = p2;
         }
        
         if( (abs(bottom.x - reye_it->width/2) > abs(p.x - reye_it->width/2 ))) {
           bottom = p;
		} 
	   }
        }
	circle(frameCopy, Point(bottom.x + reye_it->x, bottom.y + reye_it->y), 3, CV_RGB(255,0,0), -1);
        circle(frameCopy, Point(top.x + reye_it->x, top.y+reye_it->y), 3, CV_RGB(255,0,0),-1);
      }
	for(leye_it = leye.begin(); leye_it != leye.end(); leye_it++) {
	rectangle(frameCopy, Point(leye_it->x, leye_it->y), Point(leye_it->x+leye_it->width, leye_it->y+leye_it->height), CV_RGB(0, 0 , 255 ));
       
	Mat edge_detected = gray(*leye_it);
	
	Mat pyr, temp;
		
	
	GaussianBlur(edge_detected, edge_detected, Size(9,9), 2.2, 2.2);
	equalizeHist(edge_detected, edge_detected);
	
        threshold(edge_detected, edge_detected, 125, 255, THRESH_BINARY);
	dilate(edge_detected, edge_detected, Mat(), Point(-1,-1),1);
	Canny(edge_detected, edge_detected, 50, 200, 3);

	threshold(edge_detected, edge_detected, 200, 255, THRESH_BINARY);
		
        dilate(edge_detected, edge_detected, Mat(), Point(-1,-1),1); 
        imshow("eyes", edge_detected);
	findContours(edge_detected, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 
	drawContours(frameCopy, contours, -1, CV_RGB(0,255,255), 1, CV_AA, hierarchy, 0, Point(leye_it->x, leye_it->y));
	imshow("contour", edge_detected);	
	
	
        int leye_contour = 0;
	
        for(int x=0; x<contours.size(); x++) {
            if(contours.at(x).size() > leye_contour) {
              leye_contour = x;
             }
         }
        for(int x=0; x<contours.at(leye_contour).size(); x++) {
        Point p = contours.at(leye_contour).at(x);
        Point p2 = contours.at(leye_contour).at(contours.at(leye_contour).size()-x-1);
      
        if(x == 0) {
         top   =  p;
         bottom = p;
        }
        else {
	 if( (abs(top.x - leye_it->width/2) > abs(p2.x - leye_it->width/2 ))) {
           top = p2;
         }
        
         if( (abs(bottom.x - leye_it->width/2) > abs(p.x - leye_it->width/2 ))) {
           bottom = p;
          } 
        }
      }
     	circle(frameCopy, Point(bottom.x + leye_it->x, bottom.y + leye_it->y), 3, CV_RGB(255,0,0), -1);
        circle(frameCopy, Point(top.x + leye_it->x, top.y+leye_it->y), 3, CV_RGB(255,0,0),-1);
      }
       
    }  
      imshow("result", frameCopy);

     if( waitKey( 10 ) >= 0 )
         goto _cleanup_;
    }
  }
 

_cleanup_:
  cvReleaseCapture( &capture );
  delete faceDetector;
  delete righteyeDetector;
  delete lefteyeDetector;
  return 0;
}

