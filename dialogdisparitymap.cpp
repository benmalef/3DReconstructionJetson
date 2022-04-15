#include "dialogdisparitymap.h"
#include "dialogsemiglobalblockmatching.h"
#include "dialogsemiglobalblockmatchingcuda.h"
#include "dialogstereobm.h"
#include "ui_dialogdisparitymap.h"

DialogDisparityMap::DialogDisparityMap(QWidget *parent)
    : QDialog(parent), ui(new Ui::DialogDisparityMap) {
  ui->setupUi(this);
  this->setWindowTitle("Disparity Map");
}

DialogDisparityMap::~DialogDisparityMap() { delete ui; }

void DialogDisparityMap::on_pushButton_clicked() { reject(); }

void DialogDisparityMap::on_btnStereoCPU_clicked() {

  DialogStereoBM *dialogStereoBM = new DialogStereoBM(this);
  dialogStereoBM->exec();
}

void DialogDisparityMap::on_btnStereoBMCuda_clicked() {
  DialogStereoBMcuda *dialogStereoBMcuda = new DialogStereoBMcuda;
  dialogStereoBMcuda->exec();
}

void DialogDisparityMap::on_btnStereoSGBMcpu_clicked() {
  DialogSemiGlobalBlockMatching *dialogSemiGlobalBlockMatching =
      new DialogSemiGlobalBlockMatching;
  dialogSemiGlobalBlockMatching->exec();
}

void DialogDisparityMap::on_btnStereoSGBMcuda_clicked() {
  //  DialogSemiGlobalBlockMatchingCuda *dialogSemiGlobalBlockMatchingCuda =
  //      new DialogSemiGlobalBlockMatchingCuda;
  //  dialogSemiGlobalBlockMatchingCuda->exec();
}
