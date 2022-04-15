#include "disparitymap.h"

DisparityMap::DisparityMap() { readFromFile(); }
DisparityMap::~DisparityMap() { destroyAllWindows(); }

void DisparityMap::readFromFile() {
  FileStorage cvFile =
      FileStorage("calibrationParamaters.xml", FileStorage::READ);
  cvFile["leftStereoMapX"] >> leftStereoMapX;
  cvFile["leftStereoMapY"] >> leftStereoMapY;
  cvFile["rightStereoMapX"] >> rightStereoMapX;
  cvFile["rightStereoMapY"] >> rightStereoMapY;
  cvFile["Q"] >> Q;
  cvFile.release();
}

void DisparityMap::stereoBMcpu() {
  std::chrono::time_point<std::chrono::system_clock> start, end;

  int numDisparities = 64;
  int blockSize = 11;
  Mat leftFrame;
  Mat rightFrame;
  Mat disparityMap, disparityMapVis;
  Mat colorsImage, pointsXYZ;
  auto cpuStereoBM = StereoBM::create(numDisparities, blockSize);

  //  while (stereoCamera.isOpened()) {
  //    stereoCamera.getCapLeft().read(leftFrame);
  //    stereoCamera.getCapRight().read(rightFrame);

  cout << "<-------StereoBM CPU------>" << endl;

  leftFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                     "CalibrationPictures/Left/imageL0.png");
  rightFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                      "CalibrationPictures/Right/imageR0.png");
  cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
  cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);

  remap(leftFrame, leftFrame, leftStereoMapX, leftStereoMapY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
  remap(rightFrame, rightFrame, rightStereoMapX, rightStereoMapY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
  start = std::chrono::system_clock::now();
  cpuStereoBM->compute(leftFrame, rightFrame, disparityMap);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  ximgproc::getDisparityVis(disparityMap, disparityMapVis, 1.0);
  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  imshow("DisparityMap", disparityMapVis);
  normalize(disparityMap, disparityMap, 0, 255, NORM_MINMAX, CV_8U);
  cv::reprojectImageTo3D(disparityMap, pointsXYZ, Q, false, CV_32F);
  // colors

  leftFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                     "CalibrationPictures/Left/imageL0.png");
  viz::writeCloud(
      "/home/benmalef/Desktop/3DReconstruction/pointCloudStereoBMcpu.ply",
      pointsXYZ, leftFrame);

  while (true) {
    char key = waitKey(1);
    if (key == 'q') {
      break;
    }
  }
}

void DisparityMap::stereoBMcuda() {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  int numDisparities = 128;
  int blockSize = 11;
  Mat leftFrame;
  Mat rightFrame;
  Mat colorsImage, pointsXYZ;
  auto bm = cuda::createStereoBM(numDisparities, blockSize);

  cout << "<-------StereoBM Cuda------>" << endl;
  //  while (stereoCamera.isOpened()) {
  //    stereoCamera.getCapLeft().read(leftFrame);
  //    stereoCamera.getCapRight().read(rightFrame);
  leftFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                     "CalibrationPictures/Left/imageL0.png");
  rightFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                      "CalibrationPictures/Right/imageR0.png");
  cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
  cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);

  remap(leftFrame, leftFrame, leftStereoMapX, leftStereoMapY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
  remap(rightFrame, rightFrame, rightStereoMapX, rightStereoMapY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);

  Mat disparityMap(leftFrame.size(), CV_16S);
  cuda::GpuMat cudaDisparityMap(leftFrame.size(), CV_16S);
  cuda::GpuMat cudaDrawColorDisparity(leftFrame.size(), CV_8UC4);

  start = std::chrono::system_clock::now();

  cudaLeftFrame.upload(leftFrame);
  cudaRightFrame.upload(rightFrame);
  bm->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);
  cudaDisparityMap.download(disparityMap);

  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  imshow("DisparityMap", (Mat_<uchar>)disparityMap);

  // colors
  leftFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                     "CalibrationPictures/Left/imageL0.png");
  cv::reprojectImageTo3D(disparityMap, pointsXYZ, Q, false, CV_32F);
  viz::writeCloud(
      "/home/benmalef/Desktop/3DReconstruction/pointCloudStereoBMcuda.ply",
      pointsXYZ, leftFrame);

  while (true) {
    char key = waitKey(1);
    if (key == 'q') {
      break;
    }
  }
}

void DisparityMap::stereoSGBMcpu() {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  Mat leftFrame;
  Mat rightFrame;
  Mat disparityMap, disparityMapVis;
  Mat colorsImage, pointsXYZ;
  int minDisparity = 0;
  int numDisparities = 96;
  int blockSize = 11;
  int P1 = 0;
  int P2 = 0;
  int disp12MaxDiff = 0;
  int preFilterCap = 0;
  int uniquenessRatio = 0;
  int speckleWindowSize = 0;
  int speckleRange = 0;
  int mode = StereoSGBM::MODE_SGBM;

  auto cpuStereoSGBM = StereoSGBM::create(
      minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff,
      preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);

  //  while (stereoCamera.isOpened()) {
  //    stereoCamera.getCapLeft().read(leftFrame);
  //    stereoCamera.getCapRight().read(rightFrame);

  cout << "<-------StereoSGBM CPU------>" << endl;

  leftFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                     "CalibrationPictures/Left/imageL0.png");
  rightFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                      "CalibrationPictures/Right/imageR0.png");
  cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
  cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);

  remap(leftFrame, leftFrame, leftStereoMapX, leftStereoMapY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
  remap(rightFrame, rightFrame, rightStereoMapX, rightStereoMapY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
  start = std::chrono::system_clock::now();
  cpuStereoSGBM->compute(leftFrame, rightFrame, disparityMap);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  ximgproc::getDisparityVis(disparityMap, disparityMapVis, 1.0);
  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  imshow("DisparityMap", disparityMapVis);
  normalize(disparityMap, disparityMap, 0, 255, NORM_MINMAX, CV_8U);
  cv::reprojectImageTo3D(disparityMap, pointsXYZ, Q, false, CV_32F);
  // colors

  cvtColor(leftFrame, leftFrame, COLOR_GRAY2BGR);
  viz::writeCloud(
      "/home/benmalef/Desktop/3DReconstruction/pointCloudStereoSGBMcpu.ply",
      pointsXYZ, leftFrame);

  while (true) {
    char key = waitKey(1);
    if (key == 'q') {
      break;
    }
  }
}

void DisparityMap::stereoSGBMcuda() {
  std::chrono::time_point<std::chrono::system_clock> start, end;

  Mat leftFrame;
  Mat rightFrame;
  Mat colorsImage, pointsXYZ;
  int minDisparity = 0;
  int numDisparities = 96;
  int blockSize = 11;
  int P1 = 0;
  int P2 = 0;
  int disp12MaxDiff = 0;
  int preFilterCap = 0;
  int uniquenessRatio = 0;
  int speckleWindowSize = 0;
  int speckleRange = 0;
  int mode = StereoSGBM::MODE_SGBM;

  auto cudaStereoSGBM = cuda::StereoSGM::create(
      minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff,
      preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
  cout << "<-------stereoSGBM Cuda------>" << endl;
  //  while (stereoCamera.isOpened()) {
  //    stereoCamera.getCapLeft().read(leftFrame);
  //    stereoCamera.getCapRight().read(rightFrame);
  leftFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                     "CalibrationPictures/Left/imageL0.png");
  rightFrame = imread("/home/benmalef/Desktop/3DReconstruction/"
                      "CalibrationPictures/Right/imageR0.png");
  cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
  cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);

  remap(leftFrame, leftFrame, leftStereoMapX, leftStereoMapY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
  remap(rightFrame, rightFrame, rightStereoMapX, rightStereoMapY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);

  Mat disparityMap(leftFrame.size(), CV_16S);
  cuda::GpuMat cudaDisparityMap(leftFrame.size(), CV_16S);
  cuda::GpuMat cudaDrawColorDisparity(leftFrame.size(), CV_8UC4);

  start = std::chrono::system_clock::now();

  cudaLeftFrame.upload(leftFrame);
  cudaRightFrame.upload(rightFrame);

  cudaStereoSGBM->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);
  cudaDisparityMap.download(disparityMap);
  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  imshow("DisparityMap", (Mat_<uchar>)disparityMap);

  // colors
  cv::reprojectImageTo3D(disparityMap, pointsXYZ, Q, false, CV_32F);
  viz::writeCloud(
      "/home/benmalef/Desktop/3DReconstruction/pointCloudStereoBMcuda.ply",
      pointsXYZ, leftFrame);

  while (true) {
    char key = waitKey(1);
    if (key == 'q') {
      break;
    }
  }
}

void DisparityMap::realTimeDisparityMap() {

  int numDisparities = 96;
  int blockSize = 5;
  Mat leftFrame;
  Mat rightFrame;
  Mat colorsImage, pointsXYZ;
  auto bm = cuda::createStereoBM(numDisparities, blockSize);
Remap myRemap;
  StereoCamera stereoCamera;
  namedWindow("disparityMap", WINDOW_AUTOSIZE);
  createTrackbar("DisparityMap","disparityMap",0,10,0,0);

  while (stereoCamera.isOpened()) {
    stereoCamera.getCapLeft().read(leftFrame);
    stereoCamera.getCapRight().read(rightFrame);

    cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
    cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);


   myRemap.remapFrames(leftFrame,leftFrame,rightFrame,rightFrame);

    cudaLeftFrame.upload(leftFrame);
    cudaRightFrame.upload(rightFrame);
    Mat disparityMap(leftFrame.size(), CV_16S);
    cuda::GpuMat cudaDisparityMap(leftFrame.size(), CV_16S);
    cuda::GpuMat cudaColorDisparityMap(leftFrame.size(), CV_16S);
    bm->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);
    cuda::drawColorDisp(cudaDisparityMap, cudaColorDisparityMap, 96);
    cudaDisparityMap.download(disparityMap);
    cudaColorDisparityMap.download(disparityMap);

    cudaLeftFrame.download(leftFrame);
    cudaRightFrame.download(rightFrame);
    Mat conFrame;
    hconcat(leftFrame,rightFrame,conFrame);
    imshow("disparityMap", disparityMap);
//    imshow("LeftFrame", leftFrame);
//    imshow("RightFrame", rightFrame);
    imshow("Images",conFrame);

    char key = waitKey(1);
    if (key == 'q') {
      break;
    }
  }
}
