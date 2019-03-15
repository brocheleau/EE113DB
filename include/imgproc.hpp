#ifndef imgproc_hpp
#define imgproc_hpp

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>

#endif /* imgproc_hpp */

using namespace cv;
using namespace std;

// blur functions
Mat applyBoxBlur(Mat& input, const int MAX_KERNEL_LENGTH, int target_x, int target_y, int target_length, int target_height);

// color functions
Mat equalizeIntensity(const Mat& input);
Mat linearContrast(Mat& input, double alpha, double beta);

// display functions (mostly for debugging purposes)
void display1(const Mat& im1, string windowName);
void display2(const Mat& im1, const Mat& im2, string windowName);

// other useful functions
