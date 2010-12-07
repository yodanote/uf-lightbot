/**
 * @brief Object Detector Virtual Class
 * @author Robert Kirchgessner
 */

#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

class ObjectDetector
{
 public:
  virtual int detect(cv::Mat& image, std::vector<cv::Rect>& objects, cv::Rect ROI = cv::Rect()) = 0;

};

#endif /*OBJECTDETECTOR_H*/
