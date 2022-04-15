#ifndef DIALOGDISPARITYMAP_H
#define DIALOGDISPARITYMAP_H

#include "dialogstereobmcuda.h"
#include <QDialog>

namespace Ui {
class DialogDisparityMap;
}

class DialogDisparityMap : public QDialog {
  Q_OBJECT

public:
  explicit DialogDisparityMap(QWidget *parent = nullptr);
  ~DialogDisparityMap();

private slots:
  void on_btnStereoSGBMcuda_clicked();

private slots:
  void on_btnStereoSGBMcpu_clicked();

private slots:
  void on_btnStereoBMCuda_clicked();

private slots:
  void on_btnStereoCPU_clicked();

private slots:
  void on_pushButton_clicked();

private:
  Ui::DialogDisparityMap *ui;
};

#endif // DIALOGDISPARITYMAP_H
