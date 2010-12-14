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
#include "FaceDetector.h"
#include "fann.h"

using namespace std;
using namespace cv;

FaceDetector::FaceDetector(string face_cascade_file, string mouth_cascade_file, string nose_cascade_file, string left_eye_cascade_file, string right_eye_cascade_file)
{
  this->detector             = new HaarCascadeObjectDetector(face_cascade_file);
  this->mouthFeatureDetector = new MouthFeatureDetector(mouth_cascade_file);
  this->noseFeatureDetector  = new NoseFeatureDetector(nose_cascade_file);
  this->eyeFeatureDetector   = new EyeFeatureDetector(left_eye_cascade_file, right_eye_cascade_file);
  this->browFeatureDetector  = new BrowFeatureDetector();

  this->ann = fann_create_from_file("emotions.net");
}

FaceDetector::FaceDetector(cv::CascadeClassifier face_cascade, cv::CascadeClassifier mouth_cascade, cv::CascadeClassifier nose_cascade,  cv::CascadeClassifier left_cascade,  cv::CascadeClassifier right_cascade)
{
  this->detector             = new HaarCascadeObjectDetector(face_cascade);
  this->mouthFeatureDetector = new MouthFeatureDetector(mouth_cascade);
  this->noseFeatureDetector  = new NoseFeatureDetector(nose_cascade);
  this->eyeFeatureDetector   = new EyeFeatureDetector(left_cascade, right_cascade);
  this->browFeatureDetector  = new BrowFeatureDetector();

  this->ann = fann_create_from_file("emotions.net");
}

FaceDetector::~FaceDetector()
{
  delete this->detector;
  delete this->mouthFeatureDetector;
  delete this->noseFeatureDetector;
  delete this->eyeFeatureDetector;
  delete this->browFeatureDetector;

  fann_destroy(this->ann);
}



int 
FaceDetector::detect(Mat image, vector<Face>& faces)
{
  Mat gray;
  vector<Rect> face_rects;
  Face tmp;

  cvtColor(image, gray, CV_BGR2GRAY);
  detector->detect(gray, face_rects); /*Detect using Haar cascades*/ 

  /*Do feature detection*/
  vector<Rect>::const_iterator it;
  for(it = face_rects.begin(); it != face_rects.end(); it++) {
    tmp.face_box = *it;
    this->mouthFeatureDetector->detect(gray, *it, tmp.features.mouth);  /*Detect features on mouth*/
    this->noseFeatureDetector->detect(gray, *it, tmp.features.nose);    /*Detect features on nose*/
    this->eyeFeatureDetector->detect(gray, *it, tmp.features.eyes);     /*Detect eye features*/
    this->browFeatureDetector->detect(gray, *it, tmp.features.brows);   /*Detect brow features*/
    //cout<<tmp.features.mouth.lip_left_edge<<endl;
    faces.push_back(tmp);
  }
}

#if 0

#endif 

void FaceDetector::detect_emotion(DistanceFeatures emotion, DistanceFeatures neutral, double *output)
{
  fann_type *calc_out;
  fann_type input[10];

  input[0] = emotion.mouth_w/neutral.mouth_w;
  input[1] = emotion.mouth_h/neutral.mouth_h;

  input[2] = emotion.d_left_eye/neutral.d_left_eye;
  input[3] = emotion.d_right_eye/neutral.d_right_eye;

  input[4] = emotion.d_left_brow_left/neutral.d_left_brow_left;
  input[5] = emotion. d_left_brow_middle/neutral. d_left_brow_middle;
  input[6] = emotion.d_left_brow_right/neutral.d_left_brow_right;

  input[7] = emotion.d_right_brow_left/neutral.d_right_brow_left;
  input[8] = emotion.d_right_brow_middle/neutral.d_right_brow_middle;
  input[9] = emotion.d_right_brow_right/neutral.d_right_brow_right;
/*
  input[0] = 1.0;
  input[1] = 1.0;
  input[2] = 1.0;
  input[3] = 1.0;
  input[4] = 1.0;
  input[5] = 1.0;
  input[6] = 1.0;
  input[7] = 1.0;
  input[8] = 1.0;
  input[9] = 1.0;
*/
  calc_out = fann_run(this->ann, input);
  
  output[0] = calc_out[0];
  output[1] = calc_out[1];
  output[2] = calc_out[2];
  output[3] = calc_out[3];

}



#ifdef __TEST__
int 
main(int argc, char *argv[])
{
  return 0;
}
#endif

