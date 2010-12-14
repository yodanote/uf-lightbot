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

#ifndef EYEFEATUREDETECTOR_H
#define EYEFEATUREDETECTOR_H

#include <vector>
#include <opencv2/objdetect/objdetect.hpp>
#include "ObjectDetector.h"
#include "FaceFeatures.h"

class EyeFeatureDetector
{
 public:
  EyeFeatureDetector(std::string left_eye_cascade_file, std::string right_eye_cascade_file);
  EyeFeatureDetector(cv::CascadeClassifier left_cascade, cv::CascadeClassifier right_cascade);
  ~EyeFeatureDetector();

  void detect(cv::Mat image, cv::Rect face, EyeFeatures &features);
 private:
  ObjectDetector *lefteyeDetector;
  ObjectDetector *righteyeDetector;
};

#endif /*EYEFEATUREDETECTOR_H*/
