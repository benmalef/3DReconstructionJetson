#include "dialogsemiglobalblockmatchingcuda.h"
#include "ui_dialogsemiglobalblockmatchingcuda.h"

DialogSemiGlobalBlockMatchingCuda::DialogSemiGlobalBlockMatchingCuda(
    QWidget *parent)
    : QDialog(parent), ui(new Ui::DialogSemiGlobalBlockMatchingCuda) {
  ui->setupUi(this);
}

DialogSemiGlobalBlockMatchingCuda::~DialogSemiGlobalBlockMatchingCuda() {
  delete ui;
}

void DialogSemiGlobalBlockMatchingCuda::SGBMcudaExecute() {
  StereoRectification stereoRectification;
  StereoCamera stereoCamera;
  Mat leftFrame;
  Mat rightFrame;
  int minDisparity = 0;
  int numDisparities = 256;
  int P1 = 10;
  int P2 = 120;
  int uniquenessRatio = 15;

  int mode = cv::cuda::StereoSGM::MODE_HH4;

  auto cudaStereoSGBM = cuda::createStereoSGM(minDisparity, numDisparities, P1,
                                              P2, uniquenessRatio, mode);

  cuda::GpuMat cudaLeftFrame, cudaRightFrame;

  while (stereoCamera.isOpened()) {
    stereoCamera.getCapLeft().read(leftFrame);
    stereoCamera.getCapRight().read(rightFrame);

    //    remap(leftFrame, leftFrame, stereoRectification.leftStereoMapX,
    //          stereoRectification.leftStereoMapY, cv::INTER_LANCZOS4,
    //          cv::BORDER_CONSTANT, 0);
    //    remap(rightFrame, rightFrame, stereoRectification.rightStereoMapX,
    //          stereoRectification.rightStereoMapY, cv::INTER_LANCZOS4,
    //          cv::BORDER_CONSTANT, 0);

    cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);
    cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);

    Mat disparityMap(leftFrame.size(), CV_16S);
    cuda::GpuMat cudaDisparityMap(leftFrame.size(), CV_16S);
    cuda::GpuMat cudaDrawColorDisparity(leftFrame.size(), CV_8UC4);

    cudaLeftFrame.upload(leftFrame);
    cudaRightFrame.upload(rightFrame);

    cudaStereoSGBM->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);

    cuda::drawColorDisp(cudaDisparityMap, cudaDrawColorDisparity, 128);
    // cudaDisparityMap.download(disparityMap);
    cudaDrawColorDisparity.download(disparityMap);
    imshow("LeftFrame", leftFrame);
    imshow("RightFrame", rightFrame);
    //    imshow("DisparityMap1", (Mat_<uchar>)disparityMap);
    //    applyColorMap(disparityMap, disparityMap, COLORMAP_RAINBOW);
    imshow("DisparityMap", disparityMap);

    char key = waitKey(1);
    if (key == 'q' || btnExit == true) {
      break;
    }
  }
}

void DialogSemiGlobalBlockMatchingCuda::on_pushButton_clicked() {
  SGBMcudaExecute();
}

void DialogSemiGlobalBlockMatchingCuda::on_pushButton_3_clicked() {
  reject();
  btnExit = true;
}
