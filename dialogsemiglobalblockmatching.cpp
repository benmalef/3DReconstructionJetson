#include "dialogsemiglobalblockmatching.h"
#include "ui_dialogsemiglobalblockmatching.h"

DialogSemiGlobalBlockMatching::DialogSemiGlobalBlockMatching(QWidget *parent)
    : QDialog(parent), ui(new Ui::DialogSemiGlobalBlockMatching) {
  ui->setupUi(this);
}

DialogSemiGlobalBlockMatching::~DialogSemiGlobalBlockMatching() { delete ui; }

void DialogSemiGlobalBlockMatching::SGBmatchingExecute() {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  StereoCamera stereoCamera;
  StereoRectification stereoRectification;
  Mat leftFrame;
  Mat rightFrame;
  Mat disparityMap, disparityMapVis;

  auto cpuStereoSGBM = StereoSGBM::create(
      minDisparity, numOfDisparities, blockSize, P1, P2, disp12MaxDiff,
      preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);

  while (stereoCamera.isOpened()) {
    stereoCamera.getCapLeft().read(leftFrame);
    stereoCamera.getCapRight().read(rightFrame);

    remap(leftFrame, leftFrame, stereoRectification.leftStereoMapX,
          stereoRectification.leftStereoMapY, cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT, 0);
    remap(rightFrame, rightFrame, stereoRectification.rightStereoMapX,
          stereoRectification.rightStereoMapY, cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT, 0);

    cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);
    cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);

    cpuStereoSGBM->setBlockSize(blockSize);
    cpuStereoSGBM->setDisp12MaxDiff(disp12MaxDiff);
    cpuStereoSGBM->setMinDisparity(minDisparity);
    cpuStereoSGBM->setP1(P1);
    cpuStereoSGBM->setNumDisparities(numOfDisparities);
    cpuStereoSGBM->setPreFilterCap(preFilterCap);
    cpuStereoSGBM->setSpeckleRange(speckleRange);

    cpuStereoSGBM->compute(leftFrame, rightFrame, disparityMap);
    ximgproc::getDisparityVis(disparityMap, disparityMapVis, 1.0);
    imshow("LeftFrame", leftFrame);
    imshow("RightFrame", rightFrame);
    imshow("DisparityMap", disparityMapVis);
    char key = waitKey(1);
    if (key == 'q' || this->btnExit == true) {
      break;
    }
  }
}

void DialogSemiGlobalBlockMatching::SGBMcudaExecute() {
  StereoRectification stereoRectification;
  StereoCamera stereoCamera;
  Mat leftFrame;
  Mat rightFrame;

  auto cpuStereoSGBM = cuda::createStereoSGM(minDisparity, numOfDisparities, P1,
                                             P2, uniquenessRatio, mode);

  cuda::GpuMat cudaLeftFrame, cudaRightFrame;

  while (stereoCamera.isOpened()) {
    stereoCamera.getCapLeft().read(leftFrame);
    stereoCamera.getCapRight().read(rightFrame);

    remap(leftFrame, leftFrame, stereoRectification.leftStereoMapX,
          stereoRectification.leftStereoMapY, cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT, 0);
    remap(rightFrame, rightFrame, stereoRectification.rightStereoMapX,
          stereoRectification.rightStereoMapY, cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT, 0);

    cpuStereoSGBM->setBlockSize(blockSize);
    cpuStereoSGBM->setDisp12MaxDiff(disp12MaxDiff);
    cpuStereoSGBM->setMinDisparity(minDisparity);
    cpuStereoSGBM->setP1(P1);
    cpuStereoSGBM->setNumDisparities(numOfDisparities);
    cpuStereoSGBM->setPreFilterCap(preFilterCap);
    cpuStereoSGBM->setSpeckleRange(speckleRange);

    cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);
    cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);

    Mat disparityMap(leftFrame.size(), CV_16S);
    cuda::GpuMat cudaDisparityMap(leftFrame.size(), CV_16S);
    cuda::GpuMat cudaDrawColorDisparity(leftFrame.size(), CV_8UC4);

    cudaLeftFrame.upload(leftFrame);
    cudaRightFrame.upload(rightFrame);

    cpuStereoSGBM->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);

    cuda::drawColorDisp(cudaDisparityMap, cudaDrawColorDisparity,
                        numOfDisparities);
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

void DialogSemiGlobalBlockMatching::on_btnExit_clicked() {
  btnExit = true;
  reject();
}

void DialogSemiGlobalBlockMatching::on_btnStart_clicked() {
  SGBmatchingExecute();
}

void DialogSemiGlobalBlockMatching::on_sliderMinDisparity_valueChanged(
    int value) {
  minDisparity = value;
  ui->labelMinDisparityValue->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderNumOfDisparities_valueChanged(
    int value) {
  numOfDisparities = value;
  ui->labelNumOfDisparitiesValue->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderPreFilterCap_valueChanged(
    int value) {
  preFilterCap = value;
  ui->labelPreFilterCapValue->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderSpeckleRange_valueChanged(
    int value) {
  speckleRange = value;
  ui->labelSpeckleRangeValue->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderSpeckleWindowSize_valueChanged(
    int value) {
  speckleWindowSize = value;
  ui->labelSpeckleWindowSizeValue->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderDisp12MaxDiff_valueChanged(
    int value) {
  disp12MaxDiff = value;
  ui->labelDisp12MaxDiffValue->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderUniquessRatio_valueChanged(
    int value) {
  uniquenessRatio = value;
  ui->labelUniquenessRatioValue->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderP1_valueChanged(int value) {
  P1 = value;
  ui->labelP1Value->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderP2_valueChanged(int value) {
  P2 = value;
  ui->labelP2Value->setText(QString::number(value));
}

void DialogSemiGlobalBlockMatching::on_sliderBlockSize_valueChanged(int value) {
  blockSize = value;
  ui->labelBlockSizeValue->setText(QString::number(value));
}
