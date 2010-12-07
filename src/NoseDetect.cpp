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
  ObjectDetector *noseDetector = new HaarCascadeObjectDetector("haarcascade_mcs_nose.xml");
  ((HaarCascadeObjectDetector*)noseDetector)->setImageScale(1.0);

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
      for(it = faces.begin(); it != faces.end(); it++) {
        rectangle(frameCopy, Point(it->x, it->y), Point(it->x+it->width, it->y+it->height), CV_RGB(255, 0 , 0));
     
        Rect noseROI(it->x + cvRound(it->width/4.0), it->y + cvRound(it->height/2.0), cvRound(2.0*it->width/4.0),cvRound(it->height/2.0));
 
        //rectangle(frameCopy, Point(noseROI.x, noseROI.y), Point(noseROI.x + noseROI.width, noseROI.y + noseROI.height), CV_RGB(0, 255, 0));
        vector<Rect> noses;
        noseDetector->detect(gray, noses, noseROI);

        vector<Rect>::const_iterator nose_it;


        cvNamedWindow("nose", 1);
        for(nose_it = noses.begin(); nose_it != noses.end(); nose_it++) {
         // rectangle(frameCopy, Point(eyes_it->x, eyes_it->y), Point(eyes_it->x+eyes_it->width, eyes_it->y+eyes_it->height), CV_RGB(0, 0 , 255 ));
          Point center(nose_it->x + cvRound(nose_it->width/2.0), nose_it->y + cvRound(nose_it->height/2.0));
          circle(frameCopy, center, 3,  CV_RGB(255, 0 , 0 ), -1);
 
          Mat edge_detected = gray(noseROI);
          equalizeHist( edge_detected,edge_detected );
         //GaussianBlur( edge_detected, edge_detected, Size(9, 9), 2, 2 );
          
          imshow("nose", edge_detected);
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
  delete noseDetector;
  return 0;
}

