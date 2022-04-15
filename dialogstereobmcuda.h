#ifndef DIALOGSTEREOBMCUDA_H
#define DIALOGSTEREOBMCUDA_H

#include "stereocamera.h"
#include "stereorectification.h"
#include <QDialog>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

namespace Ui {
class DialogStereoBMcuda;
}

class DialogStereoBMcuda : public QDialog {
  Q_OBJECT

public:
  explicit DialogStereoBMcuda(QWidget *parent = nullptr);
  ~DialogStereoBMcuda();
  void stereoBMcudaExecute();

  void guassianFilter();
  void readImage();
  void validCloudPoint(const Mat &pointsXYZ);
  void validCloudPoint();

private slots:
  void on_btn3Dimage_clicked();
  void on_sliderBlockSize_valueChanged(int value);
  void on_sliderDisparityNum_valueChanged(int value);
  void on_pushButton_clicked();
  void on_btnSave_clicked();
  void on_btnExit_clicked();

private:
  Ui::DialogStereoBMcuda *ui;
  int numDisparities = 16;
  int blockSize = 5;
  bool btnExit = false;
  Mat pointsXYZ;
};

#endif // DIALOGSTEREOBMCUDA_H
