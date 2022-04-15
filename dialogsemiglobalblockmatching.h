#ifndef DIALOGSEMIGLOBALBLOCKMATCHING_H
#define DIALOGSEMIGLOBALBLOCKMATCHING_H

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
class DialogSemiGlobalBlockMatching;
}

class DialogSemiGlobalBlockMatching : public QDialog {
  Q_OBJECT

public:
  explicit DialogSemiGlobalBlockMatching(QWidget *parent = nullptr);
  ~DialogSemiGlobalBlockMatching();
  void SGBmatchingExecute();

  void SGBMcudaExecute();
private slots:
  void on_sliderBlockSize_valueChanged(int value);

private slots:
  void on_sliderP2_valueChanged(int value);

private slots:
  void on_sliderP1_valueChanged(int value);

private slots:
  void on_sliderUniquessRatio_valueChanged(int value);

private slots:
  void on_sliderDisp12MaxDiff_valueChanged(int value);

private slots:
  void on_sliderSpeckleWindowSize_valueChanged(int value);

private slots:
  void on_sliderSpeckleRange_valueChanged(int value);

private slots:
  void on_sliderPreFilterCap_valueChanged(int value);

private slots:
  void on_sliderNumOfDisparities_valueChanged(int value);

private slots:
  void on_sliderMinDisparity_valueChanged(int value);

private slots:
  void on_btnStart_clicked();

private slots:
  void on_btnExit_clicked();

private:
  Ui::DialogSemiGlobalBlockMatching *ui;
  int minDisparity = 0; // normally is 0
  int numOfDisparities = 256;
  int blockSize = 5; // 3-11 range, must be odd and >1
  int P1 = 0; // 8*number_of_image_channels*blockSize*blockSize // controlling
              // the smoothness
  int P2 = 0; // 32*number_of_image_channels*blockSize*blockSize
  int disp12MaxDiff = 0;
  int preFilterCap = 0;       //
  int uniquenessRatio = 5;    // 5-15 range
  int speckleWindowSize = 50; // 50-200 range
  int speckleRange = 1;       // 1-2
  int mode = cv::cuda::StereoSGM::MODE_HH4;

  bool btnExit = false;
};

#endif // DIALOGSEMIGLOBALBLOCKMATCHING_H
