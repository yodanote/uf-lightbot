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

#ifndef NOSEFEATUREDETECTOR_H
#define NOSEFEATUREDETECTOR_H

#include <vector>
#include <opencv2/objdetect/objdetect.hpp>

#include "ObjectDetector.h"
#include "FaceFeatures.h"

class NoseFeatureDetector
{
 public:
  NoseFeatureDetector(std::string cascade_file);
  NoseFeatureDetector(cv::CascadeClassifier cascade);
  ~NoseFeatureDetector();

  int detect(cv::Mat image, cv::Rect face, NoseFeatures &nose);
 private:
  ObjectDetector *detector;  
};

#endif /*NOSEFEATUREDETECTOR_H*/
