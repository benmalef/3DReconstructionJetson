#ifndef DIALOGREMAP_H
#define DIALOGREMAP_H

#include "visualizerstereocamera.h"
#include <QDialog>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

namespace Ui {
class DialogRemap;
}

class DialogRemap : public QDialog {
  Q_OBJECT

public:
  explicit DialogRemap(QWidget *parent = nullptr);
  ~DialogRemap();

  void remapShow();
  void saveImages(const Mat &frameLeft, const Mat &frameRight);
private slots:
  void on_btnScreenShot_clicked();

private slots:
  void on_btnStart_clicked();

private slots:
  void on_btnExit_clicked();

private:
  Ui::DialogRemap *ui;
  VisualizerStereoCamera *visualizerStereoCamera;
  bool btnExit = false;
  bool btnScreenShoot = false;
};

#endif // DIALOGREMAP_H
