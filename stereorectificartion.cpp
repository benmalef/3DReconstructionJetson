#include "stereorectificartion.h"

StereoRectificartion::StereoRectificartion() {}

void StereoRectificartion::readParametersFromFile() {
  FileStorage cvFile =
      FileStorage("calibrationParamaters.xml", FileStorage::READ);
  cvFile["leftStereoMapX"] >> leftStereoMapX;
  cvFile["leftStereoMapY"] >> leftStereoMapY;
  cvFile["rightStereoMapX"] >> rightStereoMapX;
  cvFile["rightStereoMapY"] >> rightStereoMapY;
  cvFile["Q"] >> Q;
  cvFile.release();
}
