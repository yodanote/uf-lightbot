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
#include "FaceFeatures.h"
#include "FaceDetector.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void process_features(Mat &frame, Face face);

int
main(int argc, char *argv[])
{
  CvCapture* capture = NULL;

  double t = 0.0;
  Mat frame, frameCopy;

  Mat gray;
  string fname;
  vector<Face> faces;

  unsigned int kmod = 0;

  FaceDetector faceDetector("haarcascade_frontalface_alt.xml", 
                            "haarcascade_mcs_mouth.xml",
                            "haarcascade_mcs_nose.xml");

  capture = cvCaptureFromCAM(0);
  cvNamedWindow("result", 1);

  if( capture ) {
    cout << "In capture ..." << endl;
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
  
      if( waitKey( 10 ) >= 0 )
        goto _cleanup_;

    if(kmod%2 == 0) {
      faces.clear(); /*Empty face list*/
      faceDetector.detect(frameCopy, faces);
    }
    vector<Face>::const_iterator it;
    for(it = faces.begin(); it != faces.end(); it++) {
      process_features(frameCopy, *it);
    }
    imshow("result", frameCopy);
   }
  }
 

_cleanup_:
  cvReleaseCapture( &capture );

  return 0;
}

void 
process_features(Mat &frame, Face face)
{
  cout<<"processing"<<endl;
  rectangle(frame, Point(face.face_box.x, face.face_box.y), Point(face.face_box.x+ face.face_box.width, face.face_box.y+face.face_box.height), CV_RGB(0, 0 , 255 ));   
         
  circle(frame, face.features.mouth.lip_left_edge, 3, CV_RGB(255, 0 , 0 ), -1);
  circle(frame, face.features.mouth.lip_right_edge, 3, CV_RGB(255, 0 , 0 ), -1);
  circle(frame, face.features.mouth.lip_top_center, 3, CV_RGB(255, 0 , 0 ), -1);
  circle(frame, face.features.mouth.lip_bottom_center, 3, CV_RGB(255, 0 , 0 ), -1);

  circle(frame, face.features.nose.center, 3, CV_RGB(0, 255 , 0 ), -1);
}

