#include "dialogremap.h"
#include "stereocamera.h"
#include "ui_dialogremap.h"

DialogRemap::DialogRemap(QWidget *parent)
    : QDialog(parent), ui(new Ui::DialogRemap) {
  ui->setupUi(this);
}

DialogRemap::~DialogRemap() { delete ui; }

void DialogRemap::on_btnExit_clicked() {
  reject();
  btnExit = true;
}

void DialogRemap::remapShow() {
  StereoCamera stereoCamera;
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
  Mat concatImage;
  Mat remapFrameLeft;
  Mat remapFrameRight;
  while (stereoCamera.isOpened()) {
    btnScreenShoot = false;
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
    if (btnScreenShoot == true) {
      saveImages(frameLeft, frameRight);
    }
    if (k == 'q' || btnExit == true) {
      break;
    }
  }
}

void DialogRemap::on_btnStart_clicked() { remapShow(); }

void DialogRemap::saveImages(const Mat &frameLeft, const Mat &frameRight) {

  String pathName = "/home/benmalef/Desktop/StereoApp/StereoImages";
  imwrite(pathName + "/screenShotR" + ".png", frameRight);
  imwrite(pathName + "/screenShotL" + ".png", frameLeft);
}

void DialogRemap::on_btnScreenShot_clicked() { btnScreenShoot = true; }
