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
#include <cmath>
#include "FaceFeatures.h"
#include "FaceDetector.h"
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>

using namespace std;
using namespace cv;

void process_features(FaceDetector &face_detect,Mat &frame, Face face);

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
                            "haarcascade_mcs_nose.xml",
                            "haarcascade_mcs_lefteye.xml",
                            "haarcascade_mcs_righteye.xml");

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
      process_features(faceDetector, frameCopy, *it);
    }
    imshow("result", frameCopy);
   }
  }
 

_cleanup_:
  cvReleaseCapture( &capture );

  return 0;
}

double calc_dist(Point p1, Point p2)
{
  double v;
  v = sqrt(pow((p1.x-p2.x),2) + pow((p1.y-p2.y),2));
  return v;
}

void 
process_features(FaceDetector &face_detect, Mat &frame, Face face)
{
  static bool initialized = false;
  static DistanceFeatures neutral;
  static int init_counter = 0;

 // cout<<"processing"<<endl;
  rectangle(frame, Point(face.face_box.x, face.face_box.y), Point(face.face_box.x+ face.face_box.width, face.face_box.y+face.face_box.height), CV_RGB(0, 0 , 255 ));   
         
  circle(frame, face.features.mouth.lip_left_edge, 3, CV_RGB(255, 0 , 0 ), -1);
  circle(frame, face.features.mouth.lip_right_edge, 3, CV_RGB(255, 0 , 0 ), -1);
  circle(frame, face.features.mouth.lip_top_center, 3, CV_RGB(255, 0 , 0 ), -1);
  circle(frame, face.features.mouth.lip_bottom_center, 3, CV_RGB(255, 0 , 0 ), -1);

  circle(frame, face.features.eyes.left_eye_top, 3, CV_RGB(255, 0 , 0 ), -1);
  circle(frame, face.features.eyes.left_eye_bottom, 3, CV_RGB(255, 0 , 0 ), -1);

  circle(frame, face.features.eyes.right_eye_top, 3, CV_RGB(255, 0 , 0 ), -1);
  circle(frame, face.features.eyes.right_eye_bottom, 3, CV_RGB(255, 0 , 0 ), -1);

  circle(frame, face.features.brows.left_brow_left, 3, CV_RGB(0, 255 , 0 ), -1);
  circle(frame, face.features.brows.left_brow_center, 3, CV_RGB(0, 255 , 0 ), -1);
  circle(frame, face.features.brows.left_brow_right, 3, CV_RGB(0, 255 , 0 ), -1);
  
  circle(frame, face.features.brows.right_brow_left, 3, CV_RGB(0, 255 , 0 ), -1);
  circle(frame, face.features.brows.right_brow_center, 3, CV_RGB(0,255 , 0 ), -1);
  circle(frame, face.features.brows.right_brow_right, 3, CV_RGB(0, 255 , 0 ), -1);

  circle(frame, face.features.nose.center, 3, CV_RGB(0, 255 , 0 ), -1);

  double mouth_w = calc_dist(face.features.mouth.lip_left_edge, face.features.mouth.lip_right_edge);
  double mouth_h = calc_dist(face.features.mouth.lip_top_center, face.features.mouth.lip_bottom_center);
  
  double d_left_eye  = calc_dist(face.features.eyes.left_eye_top, face.features.eyes.left_eye_bottom); 
  double d_right_eye = calc_dist(face.features.eyes.right_eye_top, face.features.eyes.right_eye_bottom);
  
  double d_left_brow_left   = calc_dist(face.features.brows.left_brow_left, face.features.nose.center);
  double d_left_brow_middle = calc_dist(face.features.brows.left_brow_center, face.features.nose.center);
  double d_left_brow_right  = calc_dist(face.features.brows.left_brow_right, face.features.nose.center);

  double d_right_brow_left   = calc_dist(face.features.brows.right_brow_left, face.features.nose.center);
  double d_right_brow_middle = calc_dist(face.features.brows.right_brow_center, face.features.nose.center);
  double d_right_brow_right  = calc_dist(face.features.brows.right_brow_right, face.features.nose.center);

  DistanceFeatures tmp;
  tmp.mouth_w = mouth_w;
  tmp.mouth_h = mouth_h;
  tmp.d_left_eye = d_left_eye;
  tmp.d_right_eye = d_right_eye;

#if 0
  tmp.d_left_brow_left   = d_left_brow_left;
  tmp.d_left_brow_middle = d_left_brow_middle;
  tmp.d_left_brow_right  = d_left_brow_right;

  tmp.d_right_brow_left  = d_right_brow_left;
  tmp.d_right_brow_middle = d_right_brow_middle;
  tmp.d_right_brow_right = d_right_brow_right;
#endif 

  if(init_counter == 0) {
    cout<<"Calibrating neutral face.."<<endl;
    init_counter++;
  }
  else if(init_counter == 20) {
    initialized = true;
    neutral = tmp;
    init_counter++;

    cout<<"Calibrated!"<<endl;
  }
  else {
    init_counter++;
  }
  
  double out[4];

  if(initialized) {
    face_detect.detect_emotion(tmp, neutral, out);
    for(int i=0;i<4;i++) {cout<<out[i]<<" ";}
    cout<<endl;
  }
}


