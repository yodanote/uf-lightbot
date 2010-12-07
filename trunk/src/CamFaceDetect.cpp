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

  ObjectDetector *detector = new HaarCascadeObjectDetector("haarcascade_frontalface_alt.xml");

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

    if(kmod%2 == 0) {
      faces.clear();
      cvtColor(frameCopy, gray, CV_BGR2GRAY);
      cvNamedWindow("result", 1);
  
      t = (double)cvGetTickCount();
      detector->detect(gray, faces);
      t = (double)cvGetTickCount() - t;
      cout<<"execution time = "<<(t/((double)cvGetTickFrequency()*1000.0))<<" ms"<<endl;\
    }
      vector<Rect>::const_iterator it;
      for(it = faces.begin(); it != faces.end(); it++) {
        rectangle(frameCopy, Point(it->x, it->y), Point(it->x+it->width, it->y+it->height), CV_RGB(255, 0 , 0));
      }  
      imshow("result", frameCopy);

     if( waitKey( 10 ) >= 0 )
         goto _cleanup_;
    }
  }
 

_cleanup_:
  cvReleaseCapture( &capture );
  delete detector;

  return 0;
}

