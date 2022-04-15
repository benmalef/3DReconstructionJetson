#include "dialogstereobmcuda.h"
#include "ui_dialogstereobmcuda.h"

DialogStereoBMcuda::DialogStereoBMcuda(QWidget *parent)
    : QDialog(parent), ui(new Ui::DialogStereoBMcuda) {
  ui->setupUi(this);
  ui->labelDisparity->setText(QString::number(numDisparities));
  ui->labelBlockSize->setText(QString::number(blockSize));
}

DialogStereoBMcuda::~DialogStereoBMcuda() { delete ui; }

void DialogStereoBMcuda::on_btnExit_clicked() {
  reject();
  btnExit = true;
}

void DialogStereoBMcuda::readImage() {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  cuda::GpuMat cudaLeftFrame, cudaRightFrame;
  StereoRectification stereoRectification;
  Mat leftFrame;
  Mat rightFrame;
  Mat colorsImage;
  Mat disparityMapColor;
  auto bm = cuda::createStereoBM(numDisparities, blockSize);
  bm->setROI1(stereoRectification.RoiLeft);
  bm->setROI2(stereoRectification.RoiRight);

  cout << "<-------StereoBM GPU------>" << endl;
  leftFrame =
      imread("/home/benmalef/Desktop/StereoApp/StereoImages/screenShotL.png");
  rightFrame =
      imread("/home/benmalef/Desktop/StereoApp/StereoImages/screenShotR.png");

  cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
  cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);

  //  remap(leftFrame, leftFrame, stereoRectification.leftStereoMapX,
  //        stereoRectification.leftStereoMapY, cv::INTER_LANCZOS4,
  //        cv::BORDER_CONSTANT, 0);
  //  remap(rightFrame, rightFrame, stereoRectification.rightStereoMapX,
  //        stereoRectification.rightStereoMapY, cv::INTER_LANCZOS4,
  //        cv::BORDER_CONSTANT, 0);

  Mat disparityMap(leftFrame.size(), CV_16S);
  cuda::GpuMat cudaDisparityMap(leftFrame.size(), CV_16S);
  cuda::GpuMat cudaDrawColorDisparity(leftFrame.size(), CV_8UC4);

  start = std::chrono::system_clock::now();

  cudaLeftFrame.upload(leftFrame);
  cudaRightFrame.upload(rightFrame);

  bm->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);

  cuda::drawColorDisp(cudaDisparityMap, cudaDrawColorDisparity, numDisparities);

  cudaDrawColorDisparity.download(disparityMapColor);
  cudaDisparityMap.download(disparityMap);

  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  //  imshow("DisparityMap", (Mat_<uchar>)disparityMap);
  auto DispRect = getValidDisparityROI(
      stereoRectification.RoiLeft, stereoRectification.RoiRight,
      bm->getMinDisparity(), bm->getNumDisparities(), bm->getBlockSize());

  disparityMap = disparityMap(DispRect);
  disparityMapColor = disparityMapColor(DispRect);
  leftFrame = leftFrame(DispRect);
  imshow("DisparityMap", (Mat_<uchar>)disparityMap);
  imshow("DisparityMapColor", disparityMapColor);

  // colors
  cv::reprojectImageTo3D(disparityMap, pointsXYZ, stereoRectification.Q, false,
                         CV_32F);

  viz::writeCloud("/home/benmalef/Desktop/StereoApp/pointCloudStereoBMcuda.ply",
                  pointsXYZ, leftFrame);

  while (true) {
    char key = waitKey(1);
    if (key == 'q' || btnExit == true) {
      break;
    }
  }
}

void DialogStereoBMcuda::on_btnSave_clicked() {
  this->btnExit = true;
  int tmpValue = ui->sliderDisparityNum->value();
  int tmpBlockSize = ui->sliderBlockSize->value();
  if (tmpValue % 16 == 0)
    numDisparities = tmpValue;
  if (tmpBlockSize % 2 == 1)
    blockSize = tmpBlockSize;
}

void DialogStereoBMcuda::stereoBMcudaExecute() {
  StereoRectification stereoRectification;
  StereoCamera stereoCamera;
  Mat leftFrame;
  Mat rightFrame;
  Mat disparityMapColor;
  auto bm = cuda::createStereoBM(numDisparities, blockSize);
  cuda::GpuMat cudaLeftFrame, cudaRightFrame;

  while (stereoCamera.isOpened()) {
    stereoCamera.getCapLeft().read(leftFrame);
    stereoCamera.getCapRight().read(rightFrame);

    remap(leftFrame, leftFrame, stereoRectification.leftStereoMapX,
          stereoRectification.leftStereoMapY, cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT);
    remap(rightFrame, rightFrame, stereoRectification.rightStereoMapX,
          stereoRectification.rightStereoMapY, cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT);

    cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);
    cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);

    Mat disparityMap(leftFrame.size(), CV_16S);
    cuda::GpuMat cudaDisparityMap(leftFrame.size(), CV_16S);
    cuda::GpuMat cudaDrawColorDisparity(leftFrame.size(), CV_8UC4);

    cudaLeftFrame.upload(leftFrame);
    cudaRightFrame.upload(rightFrame);

    bm->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);
    cuda::drawColorDisp(cudaDisparityMap, cudaDrawColorDisparity,
                        numDisparities);
    cudaDisparityMap.download(disparityMap);
    cudaDrawColorDisparity.download(disparityMapColor);

    getValidDisparityROI(stereoRectification.RoiLeft,
                         stereoRectification.RoiRight, bm->getMinDisparity(),
                         bm->getNumDisparities(), bm->getBlockSize());

    imshow("LeftFrame", leftFrame);
    imshow("RightFrame", rightFrame);

    //    imshow("DisparityMap", (Mat_<uchar>)disparityMap);
    //    applyColorMap(disparityMap, disparityMap, COLORMAP_RAINBOW);
    imshow("DisparityMap", disparityMapColor);

    char key = waitKey(1);
    if (key == 'q' || btnExit == true) {
      break;
    }
  }
}

void DialogStereoBMcuda::on_pushButton_clicked() {
  btnExit = false;
  stereoBMcudaExecute();
}

void DialogStereoBMcuda::on_sliderDisparityNum_valueChanged(int value) {
  if (value % 16 == 0)
    ui->labelDisparity->setText(QString::number(value));
}

void DialogStereoBMcuda::on_sliderBlockSize_valueChanged(int value) {
  if (value % 2 == 1)
    ui->labelBlockSize->setText(QString::number(value));
}

void DialogStereoBMcuda::on_btn3Dimage_clicked() {
  btnExit = false;
  readImage();
}
