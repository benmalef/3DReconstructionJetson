#include "dialogstereobm.h"
#include "ui_dialogstereobm.h"

DialogStereoBM::DialogStereoBM(QWidget *parent)
    : QDialog(parent), ui(new Ui::DialogStereoBM) {
  ui->setupUi(this);
  ui->labelDisparity->setText(QString::number(numDisparities));
  ui->labelBlockSize->setText(QString::number(blockSize));
}

DialogStereoBM::~DialogStereoBM() {
  delete ui;
  destroyAllWindows();
  this->btnExit = true;
}

void DialogStereoBM::stereoBMexecute() {
  Mat leftFrame;
  Mat rightFrame;
  Mat disparityMap, disparityMapVis;
  StereoCamera stereoCamera;
  StereoRectification stereoRectification;
  auto cpuStereoBM = StereoBM::create(numDisparities, blockSize);

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
    cpuStereoBM->compute(leftFrame, rightFrame, disparityMap);

    ximgproc::getDisparityVis(disparityMap, disparityMapVis, 1.0);
    normalize(disparityMap, disparityMap, 0, 255, NORM_MINMAX, CV_8U);
    imshow("LeftFrame", leftFrame);
    imshow("RightFrame", rightFrame);

    //    applyColorMap(disparityMap, disparityMap, COLORMAP_JET);
    imshow("DisparityMap", disparityMapVis);

    char key = waitKey(1);
    if (key == 'q' || this->btnExit == true) {
      break;
    }
  }
}

void DialogStereoBM::readImageStereoBMexecute() {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  Mat leftFrame;
  Mat rightFrame;
  Mat disparityMap, disparityMapVis;
  StereoRectification stereoRectification;
  Mat colorsImage, pointsXYZ;
  auto cpuStereoBM = StereoBM::create(numDisparities, blockSize);

  leftFrame =
      imread("/home/benmalef/Desktop/StereoApp/StereoImages/screenShotL.png");
  rightFrame =
      imread("/home/benmalef/Desktop/StereoApp/StereoImages/screenShotR.png");

  //  cv::resize(leftFrame, leftFrame, Size(down_width, down_height),
  //  INTER_LINEAR); cv::resize(rightFrame, rightFrame, Size(down_width,
  //  down_height),
  //             INTER_LINEAR);

  cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
  cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);

  remap(leftFrame, leftFrame, stereoRectification.leftStereoMapX,
        stereoRectification.leftStereoMapY, cv::INTER_LANCZOS4,
        cv::BORDER_CONSTANT);
  remap(rightFrame, rightFrame, stereoRectification.rightStereoMapX,
        stereoRectification.rightStereoMapY, cv::INTER_LANCZOS4,
        cv::BORDER_CONSTANT);
  start = std::chrono::system_clock::now();
  cpuStereoBM->compute(leftFrame, rightFrame, disparityMap);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  ximgproc::getDisparityVis(disparityMap, disparityMapVis, 1.0);
  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  imshow("DisparityMap", disparityMapVis);

  normalize(disparityMap, disparityMap, 0, 255, NORM_MINMAX, CV_8U);

  reprojectImageTo3D(disparityMap, pointsXYZ, stereoRectification.Q, false,
                     CV_32F);
  // colors

  cvtColor(leftFrame, leftFrame, COLOR_GRAY2BGR);

  viz::writeCloud("/home/benmalef/Desktop/StereoApp/pointCloudStereoBMcpu.ply",
                  pointsXYZ, leftFrame);

  while (true) {
    char key = waitKey(1);
    if (key == 'q' || btnExit == true) {
      break;
    }
  }
}

void DialogStereoBM::on_btnExit_clicked() {
  reject();
  destroyAllWindows();
  this->btnExit = true;
}

void DialogStereoBM::on_pushButton_clicked() {
  this->btnExit = true;
  int tmpValue = ui->sliderDisparityNum->value();
  int tmpBlockSize = ui->sliderBlockSize->value();
  if (tmpValue % 16 == 0)
    numDisparities = tmpValue;
  if (tmpBlockSize % 2 == 1)
    blockSize = tmpBlockSize;
}

void DialogStereoBM::on_sliderDisparityNum_valueChanged(int value) {
  if (value % 16 == 0)
    ui->labelDisparity->setText(QString::number(value));
}

void DialogStereoBM::on_btnStart_clicked() {
  this->btnExit = false;
  stereoBMexecute();
  //  readImageStereoBMexecute();
}

void DialogStereoBM::on_sliderBlockSize_valueChanged(int value) {
  if (value % 2 == 1)
    ui->labelBlockSize->setText(QString::number(value));
}

void DialogStereoBM::on_btn3Dimage_clicked() {
  this->btnExit = false;
  readImageStereoBMexecute();
}
