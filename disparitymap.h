#ifndef DISPARITYMAP_H
#define DISPARITYMAP_H
#include "stereocamera.h"
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "remap.h"
using namespace cv;
using namespace std;

class DisparityMap {
public:
  DisparityMap();
  ~DisparityMap();
  void stereoBMcpu();
  void stereoBMcuda();
  void showDisparityMapFromImages();

  void readFromFile();
  void stereoSGBMcpu();

  void stereoSGBMcuda();
  void realTimeDisparityMap();
private:
  cuda::GpuMat cudaLeftFrame, cudaRightFrame;
  Mat leftStereoMapX, leftStereoMapY;
  Mat rightStereoMapX, rightStereoMapY;
  Mat Q;
};

#endif // DISPARITYMAP_H
