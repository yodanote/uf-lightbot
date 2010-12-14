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

#ifndef BROWFEATUREDETECTOR_H
#define BROWFEATUREDETECTOR_H

#include <vector>
#include <opencv2/objdetect/objdetect.hpp>
#include "ObjectDetector.h"
#include "FaceFeatures.h"

class BrowFeatureDetector
{
 public:
  BrowFeatureDetector();
  ~BrowFeatureDetector();

  void detect(cv::Mat image, cv::Rect face, BrowFeatures &features);
 private:
};

#endif /*BROWFEATUREDETECTOR_H*/
