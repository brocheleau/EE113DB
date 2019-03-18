#ifndef imgproc_hpp
#define imgproc_hpp

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>

#include "objDetection.hpp"

using namespace cv;
using namespace std;

// blur functions
void applyBoxBlur(Mat& input, const int MAX_KERNEL_LENGTH, detections locations);
void removeException(Mat& input, const int MAX_KERNEL_LENGTH, int x1, int y1, int x2, int y2);
void addException(Mat& input, Mat& original, int x1, int y1, int x2, int y2);

// color functions
void equalizeIntensity(Mat& input);
void linearContrast(Mat& input, double alpha, double beta);

// display functions (mostly for debugging purposes)
void display1(const Mat& im1, string windowName);
void display2(const Mat& im1, const Mat& im2, string windowName);

#endif /* imgproc_hpp */
