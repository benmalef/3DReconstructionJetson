#include "visualizerstereocamera.h"

VisualizerStereoCamera::VisualizerStereoCamera() {}

VisualizerStereoCamera::VisualizerStereoCamera(string pathCamLeft,
                                               string pathCamRight)
    : stereoCamera(pathCamLeft, pathCamRight) {}

VisualizerStereoCamera::~VisualizerStereoCamera() {}

void VisualizerStereoCamera::show() {
  Mat frameLeft;
  Mat frameRight;
  while (stereoCamera.isOpened()) {
    stereoCamera.getCapLeft().read(frameLeft);
    stereoCamera.getCapRight().read(frameRight);
    imshow("Left Camera", frameLeft);
    imshow("Right Camera", frameRight);
    int k = waitKey(1);
    if (k == 'q') {
      break;
    }
  }
}

void VisualizerStereoCamera::drawLines() {
  //  int imageWidth = concatImage.size().width;
  //  int imageHeight = concatImage.size().height;

  //  vector<Point> startPoints, endPoints;

  //  Point startPoint(0, 0), endPoint(0, 0);
  //  int maxNumberOfPoints[10];

  //  for (int i = 0; i < maxNumberOfPoints; ++i) {
  //    startPoint = startPoint();
  //  }
}

void VisualizerStereoCamera::remapShow() {
  Mat leftStereoMapX, leftStereoMapY;
  Mat rightStereoMapX, rightStereoMapY;
  FileStorage cvFile =
      FileStorage("calibrationParamaters.xml", FileStorage::READ);
  cvFile["leftStereoMapX"] >> leftStereoMapX;
  cvFile["leftStereoMapY"] >> leftStereoMapY;
  cvFile["rightStereoMapX"] >> rightStereoMapX;
  cvFile["rightStereoMapY"] >> rightStereoMapY;
  cvFile.release();

  Mat frameLeft;
  Mat frameRight;
  Mat remapFrameLeft;
  Mat remapFrameRight;
  while (stereoCamera.isOpened()) {
    stereoCamera.getCapLeft().read(frameLeft);
    stereoCamera.getCapRight().read(frameRight);

    remap(frameLeft, frameLeft, leftStereoMapX, leftStereoMapY,
          cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
    remap(frameRight, frameRight, rightStereoMapX, rightStereoMapY,
          cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);

    hconcat(frameLeft, frameRight, concatImage);
    //    drawLines();
    imshow("Frame", concatImage);

    int k = waitKey(1);
    if (k == 'q') {
      break;
    }
  }
}
