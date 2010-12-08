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
  ObjectDetector *mouthDetector = new HaarCascadeObjectDetector("haarcascade_mcs_mouth.xml");
  ((HaarCascadeObjectDetector*)mouthDetector)->setImageScale(1.3);

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

      cvNamedWindow("gray", 1);
      cvNamedWindow("stage1",1);
      cvNamedWindow("stage3",1);
      cvNamedWindow("stage2",1);
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
     
        Rect mouthROI(it->x, it->y + cvRound(it->height/2.0),it->width, cvRound(it->height/2.0)); 
        rectangle(frameCopy, Point(mouthROI.x, mouthROI.y), Point(mouthROI.x + mouthROI.width, mouthROI.y + mouthROI.height), CV_RGB(0, 255, 0));
        vector<Rect> lips;
        mouthDetector->detect(gray, lips, mouthROI);

        vector<Rect>::const_iterator lips_it;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        //if(lips.empty()) { cout<<"empty"; } else { cout<<"lips found"<<endl; }
        cvNamedWindow("lips", 1);
        for(lips_it = lips.begin(); lips_it != lips.end(); lips_it++) {
          rectangle(frameCopy, Point(lips_it->x, lips_it->y), Point(lips_it->x+lips_it->width, lips_it->y+lips_it->height), CV_RGB(0, 0 , 255 ));   
         
          Mat edge_detected = gray(*lips_it);
          equalizeHist(edge_detected, edge_detected);

          Mat pyr;
          /*Remove some noise by down/upsampling*/
          pyrDown(edge_detected, pyr, Size(edge_detected.cols/2, edge_detected.rows/2));
          pyrUp(pyr, edge_detected, edge_detected.size());
          imshow("gray", edge_detected);

          /*Remove some additional edge noise*/
          GaussianBlur(edge_detected, edge_detected, Size(9,9), 2.2, 2.2); 
          equalizeHist(edge_detected, edge_detected); /*Equalize color histogram*/
          imshow("stage1", edge_detected);

          imshow("stage2", edge_detected);

          threshold(edge_detected, edge_detected, 125, 255, THRESH_BINARY); /*Create color blobs*/
          dilate(edge_detected, edge_detected, Mat(), Point(-1,-1),1); /*Fill in gaps*/
          imshow("lips", edge_detected);

          Canny(edge_detected, edge_detected, 50, 200, 3);
          dilate(edge_detected, edge_detected, Mat(), Point(-1,-1),2); /*Fill in gaps*/
          erode(edge_detected, edge_detected, Mat(), Point(-1,-1),1); /*Fill in gaps*/
          threshold(edge_detected, edge_detected, 200, 255, THRESH_BINARY); /*Create edge blobs*/
          
          imshow("stage3", edge_detected);

          Mat tmp = edge_detected;
          findContours(tmp, contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 
          cout<<"found "<<contours.size()<<" contours"<<endl;
          for(int x=0; x<contours.size(); x++) {
            drawContours(frameCopy, contours, x, CV_RGB(0, 255, 255 ), 1, CV_AA, hierarchy, 0, Point(lips_it->x, lips_it->y));
            cout<<"contour size: "<<contours.at(x).size()<<endl;
            for(int y=0; y<contours.at(x).size(); y++) {
              circle(frameCopy, contours.at(x).at(y), 3, CV_RGB(255, 0 , 0 ), -1);
            }
          }

          circle(frameCopy, Point(cvRound(lips_it->x), cvRound(lips_it->y+lips_it->height/2.0)), 3, CV_RGB(255, 0 , 0 ), -1);
          circle(frameCopy, Point(cvRound(lips_it->x+lips_it->width), cvRound(lips_it->y+lips_it->height/2.0)), 3, CV_RGB(255, 0 , 0 ), -1);
          
          cout<<"dist: "<<lips_it->width<<" estimated emotion feature: "<< ((lips_it->width >= 75)? "smiling":"not smiling")<<endl;
          cout<<"height: "<<lips_it->height<<endl;
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
  delete mouthDetector;
  return 0;
}

