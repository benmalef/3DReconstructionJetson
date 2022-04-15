#include "menu.h"

using namespace std;
Menu::Menu() {}

Menu::~Menu() {}

void Menu::displayMenu() {
  cout << "<-------Menu-------->" << endl;
  cout << "1->Take calibration Images" << endl;
  cout << "2->Calibrate the StereoCamera" << endl;
  cout << "3->Show Disparity Map" << endl;
  cout << "4->3D Reconstruction" << endl;
  cout << "5->Exit" << endl;
  cout << "Choice and press choice-number:" << endl;
}

void Menu::executeChoice() {
    String pathCalibrationPicturesLeftCamera = "/home/benmalef/Desktop/3DReconstruction/CalibrationPictures/Left";
    String pathCalibratinPicturesRightCamera = "/home/benmalef/Desktop/3DReconstruction/CalibrationPictures/Right";
  displayMenu();
  cin >> this->choice;

  switch (this->choice) {
  case 1:{
    cout << "<---Calibration Images--->"<< endl;
    CalibrationPictures calibrationPictures(pathCalibrationPicturesLeftCamera,
                                            pathCalibratinPicturesRightCamera);
    calibrationPictures.takeCalibrationImages();
    break;}
  case 2:{
    cout << "<--Calibration Stereo Camera--->"<< endl;
    CalibrationStereoCamera calibrationStereoCamera(pathCalibrationPicturesLeftCamera,pathCalibratinPicturesRightCamera);
    calibrationStereoCamera.executeCaibrationStereoCamera();
    break;}
  case 3:{
    cout << "Show Disparity Map"<< endl;
//    VisualizerStereoCamera visualizerStereoCamera;
//    visualizerStereoCamera.remapShow();
    DisparityMap disparityMap;
   disparityMap.stereoBMcpu();
    disparityMap.stereoBMcuda();
    disparityMap.stereoSGBMcpu();
    disparityMap.stereoSGBMcuda();
    break;}
  case 4:{
    cout << "3D Reconstruction" << endl;
//    VisualizerStereoCamera visualizerStereoCamera;
//    visualizerStereoCamera.remapShow();
    DisparityMap disp;
    disp.realTimeDisparityMap();
//    trackBarWindow trackWindow;
//    trackWindow.createTrackBarWindow();
  break;}
  case 5:{
    cout << "Exit" << endl;
    break;}
  }
}

void Menu::exe(){
  while(this->choice != 5){
      executeChoice();
  }

}
int Menu::getChoice() const { return choice; }

void Menu::setChoice(int value) { choice = value; }
