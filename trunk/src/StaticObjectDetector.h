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


/**
 * @brief Object Detector Virtual Class
 * @author Robert Kirchgessner
 */

#ifndef STATICOBJECTDETECTOR_H
#define STATICOBJECTDETECTOR_H

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "ObjectDetector.h"

class StaticObjectDetector : public ObjectDetector
{
 public:
  StaticObjectDetector(double x_percent, double y_percent, double width_percent, double height_percent);

  virtual ~StaticObjectDetector();

  int detect(cv::Mat image, std::vector<cv::Rect>& objects, cv::Rect ROI = cv::Rect());
  }

 private:
  double xp, yp, wp, hp;
};

#endif /*STATICOBJECTDETECTOR_H*/
