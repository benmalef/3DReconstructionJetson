#include "remap.h"

Remap::Remap()
{
readFromFile();
}

Remap::~Remap()
{

}
void Remap::readFromFile() {
  FileStorage cvFile =
      FileStorage("calibrationParamaters.xml", FileStorage::READ);
  cvFile["leftStereoMapX"] >> leftStereoMapX;
  cvFile["leftStereoMapY"] >> leftStereoMapY;
  cvFile["rightStereoMapX"] >> rightStereoMapX;
  cvFile["rightStereoMapY"] >> rightStereoMapY;
  cvFile.release();
}

void Remap::remapFrames(const Mat &inLeftFrame, Mat &outLeftFrame, const Mat &inRightFrame, Mat &outRightFrame){

    cv::remap(inLeftFrame, outLeftFrame, leftStereoMapX, leftStereoMapY,
          cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
    cv::remap(inRightFrame, outRightFrame, rightStereoMapX, rightStereoMapY,
          cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);

}
