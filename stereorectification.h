#ifndef STEREORECTIFICATION_H
#define STEREORECTIFICATION_H
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

class StereoRectification {
public:
  StereoRectification();
  ~StereoRectification();
  void readParametersFromFile();
  Mat leftStereoMapX, leftStereoMapY;
  Mat rightStereoMapX, rightStereoMapY;
  Mat Q;
  Rect RoiLeft, RoiRight;
};

#endif // STEREORECTIFICATION_H
