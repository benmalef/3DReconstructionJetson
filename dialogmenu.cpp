#include "dialogmenu.h"
#include "dialogremap.h"
#include "ui_dialogmenu.h"
#include "visualizerstereocamera.h"

DialogMenu::DialogMenu(QWidget *parent)
    : QDialog(parent), ui(new Ui::DialogMenu) {
  ui->setupUi(this);
  this->setWindowTitle("3D Reconstruction Menu");
}

DialogMenu::~DialogMenu() { delete ui; }

void DialogMenu::on_btnCalibrationImages_clicked() {
  Dialog *dialog = new Dialog(this);
  dialog->exec();
}

void DialogMenu::on_btnExit_clicked() { reject(); }

void DialogMenu::on_btnCalibration_clicked() {
  DialogCalibration *dialogCal = new DialogCalibration(this);
  dialogCal->exec();
}

void DialogMenu::on_btnDisparityMap_clicked() {
  DialogDisparityMap *dialogDisparityMap = new DialogDisparityMap(this);
  dialogDisparityMap->exec();
}

void DialogMenu::on_btnRemap_clicked() {
  DialogRemap *dialogRemap = new DialogRemap(this);
  dialogRemap->exec();
}
