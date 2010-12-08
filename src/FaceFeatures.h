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

#ifndef FACEFEATURES_H
#define FACEFEATURES_H

#include <opencv2/imgproc/imgproc.hpp>

typedef struct {
  cv::Point center;  /*Point representing the center of the nose*/
} NoseFeatures;

typedef struct {
  cv::Point lip_left_edge;     /*Point representing the left edge of the lip*/
  cv::Point lip_right_edge;    /*Point representing the right edge of the lip*/
  cv::Point lip_top_center;    /*Point representing the top center of the lip*/
  cv::Point lip_bottom_center; /*Point representing the bottom center of the lip*/
} MouthFeatures;

typedef struct {
  cv::Point pupil_center;      /*Point representing the pupil center*/
  cv::Point brow_left;         /*Point representing the eye-brow left point*/
  cv::Point brow_right;        /*Point representing the eye-brow right point*/
  cv::Point brow_center;       /*Point representing the eye-brow center point*/
} EyeFeatures;

typedef struct {
  NoseFeatures  nose;
  MouthFeatures mouth;
  EyeFeatures   eye;
} FaceFeatures;

typedef struct {
  cv::Rect     face_box;
  FaceFeatures features;
} Face;

#endif /*FACEFEATURES_H*/
