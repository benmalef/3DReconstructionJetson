#ifndef REMAP_H
#define REMAP_H

#include "calibrationpictures.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "remap.h"

using namespace cv;
using namespace std;

class Remap
{
public:
    Remap();
    ~Remap();

    void readFromFile();
    void remapFrames(const Mat &inLeftFrame, Mat &outLeftFrame, const Mat &inRightFrame, Mat &outRightFrame);
private:
    Mat leftStereoMapX, leftStereoMapY;
    Mat rightStereoMapX, rightStereoMapY;
};

#endif // REMAP_H
