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
                            "haarcascade_mcs_mouth.xml");

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
    imshow("result", frameCopy);
   }
  }
 

_cleanup_:
  cvReleaseCapture( &capture );

  return 0;
}
