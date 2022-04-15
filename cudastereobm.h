#ifndef CUDASTEREOBM_H
#define CUDASTEREOBM_H
#include "stereocamera.h"
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
#include "remap.h"
class cudaStereoBM
{
public:
    cudaStereoBM();
    void realTimeDisparityMap(int numDisparities, int blockSize);
    ~cudaStereoBM();
    void realTimeDisparityMap(int numDisparities, int blockSize, const string &windowName);
};

#endif // CUDASTEREOBM_H
