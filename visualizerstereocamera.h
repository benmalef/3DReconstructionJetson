#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <stdio.h>
#include "stereocamera.h"
#include "calibrationpictures.h"

using namespace cv;
using namespace std;
class VisualizerStereoCamera
{
public:
    VisualizerStereoCamera();
    ~VisualizerStereoCamera();
    VisualizerStereoCamera(string pathCamLeft, string pathCamRight);
    void drawLines();
    void show();
    void remapShow();
private:
    StereoCamera stereoCamera;
    Mat concatImage;
};

#endif // VISUALIZER_H
