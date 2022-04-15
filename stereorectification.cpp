#include "stereorectification.h"

StereoRectification::StereoRectification() { readParametersFromFile(); }
StereoRectification::~StereoRectification() { destroyAllWindows(); }
void StereoRectification::readParametersFromFile() {
  FileStorage cvFile =
      FileStorage("calibrationParamaters.xml", FileStorage::READ);
  cvFile["leftStereoMapX"] >> leftStereoMapX;
  cvFile["leftStereoMapY"] >> leftStereoMapY;
  cvFile["rightStereoMapX"] >> rightStereoMapX;
  cvFile["rightStereoMapY"] >> rightStereoMapY;
  cvFile["Q"] >> Q;
  cvFile["RoiLeftX"] >> RoiLeft.x;
  cvFile["RoiLefY"] >> RoiLeft.y;
  cvFile["RoiLeftHeight"] >> RoiLeft.height;
  cvFile["RoiLeftWidth"] >> RoiLeft.width;
  cvFile["RoiRightX"] >> RoiRight.x;
  cvFile["RoiRightY"] >> RoiRight.y;
  cvFile["RoiRightHeight"] >> RoiRight.height;
  cvFile["RoiRightWidth"] >> RoiRight.width;

  cvFile.release();
}
