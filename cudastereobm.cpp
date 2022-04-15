#include "cudastereobm.h"

cudaStereoBM::cudaStereoBM() {}
cudaStereoBM::~cudaStereoBM() {}

void cudaStereoBM::realTimeDisparityMap(int numDisparities, int blockSize, const string &windowName) {

  Mat leftFrame;
  Mat rightFrame;
  Mat colorsImage, pointsXYZ;
  auto bm = cuda::createStereoBM(numDisparities, blockSize);
  Remap remapFrames;
  Mat disparityMap(leftFrame.size(), CV_16S);
  StereoCamera stereoCamera;
  cuda::GpuMat cudaDisparityMap(leftFrame.size(), CV_16S);
  cuda::GpuMat cudaColorDisparityMap(leftFrame.size(), CV_16S);
  cuda::GpuMat cudaLeftFrame, cudaRightFrame;
  Mat conFrame;

  while (stereoCamera.isOpened()) {
    stereoCamera.getCapLeft().read(leftFrame);
    stereoCamera.getCapRight().read(rightFrame);

    cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
    cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);
    remapFrames.remapFrames(leftFrame, leftFrame, rightFrame, rightFrame);

    cudaLeftFrame.upload(leftFrame);
    cudaRightFrame.upload(rightFrame);

    bm->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);
    cuda::drawColorDisp(cudaDisparityMap, cudaColorDisparityMap, 128);
    cudaColorDisparityMap.download(disparityMap);

    cudaLeftFrame.download(leftFrame);
    cudaRightFrame.download(rightFrame);
    hconcat(leftFrame, rightFrame, conFrame);
    imshow("disparityMap", disparityMap);

    imshow(windowName, conFrame);

    char key = waitKey(1);
    if (key == 'q') {
      break;
    }
  }
}
