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

#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <vector>
#include "ObjectDetector.h"
#include "FaceFeatures.h"
#include "MouthFeatureDetector.h"
#include "NoseFeatureDetector.h"
#include "EyeFeatureDetector.h"

class FaceDetector
{
 public:
  FaceDetector(std::string face_cascade_file, std::string mouth_cascade_file);
  FaceDetector(cv::CascadeClassifier face_cascade, cv::CascadeClassifier mouth_cascade);
  ~FaceDetector();

  int detect(cv::Mat image, std::vector<Face>& faces);
 private:
  ObjectDetector *detector;
  MouthFeatureDetector *mouthFeatureDetector;
};

#endif /*FACEDETECTOR_H*/
