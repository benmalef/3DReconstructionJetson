#ifndef DIALOGSTEREOBM_H
#define DIALOGSTEREOBM_H
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
class DialogStereoBM;
}

class DialogStereoBM : public QDialog {
  Q_OBJECT

public:
  explicit DialogStereoBM(QWidget *parent = nullptr);
  ~DialogStereoBM();
  void stereoBMexecute();
  void readImageStereoBMexecute();

private slots:
  void on_btn3Dimage_clicked();

private slots:
  void on_sliderBlockSize_valueChanged(int value);

private slots:
  void on_btnStart_clicked();

private slots:
  void on_sliderDisparityNum_valueChanged(int value);

private slots:
  void on_btnExit_clicked();
  void on_pushButton_clicked();

private:
  Ui::DialogStereoBM *ui;
  int numDisparities = 32;
  int blockSize = 5;
  bool btnExit = false;
};

#endif // DIALOGSTEREOBM_H
