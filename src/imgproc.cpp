#include "imgproc.hpp"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/////////////////
// blur functions
/////////////////

Mat applyBoxBlur(Mat& input, const int MAX_KERNEL_LENGTH, int target_x, int target_y, int target_length, int target_height){

    Mat dst;
    Mat temp;
    
    temp = input.clone();
    
    for( int y = target_y + MAX_KERNEL_LENGTH; y < (target_y + target_height - MAX_KERNEL_LENGTH); y++ ) {
        for( int x = target_x + MAX_KERNEL_LENGTH; x < (target_x + target_length - MAX_KERNEL_LENGTH); x++ ) {
            for( int c = 0; c < temp.channels(); c++ ) {
                temp.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(0);
            }
        }
    }
    
    for( int i = 1 ; i < MAX_KERNEL_LENGTH ; i+=2) {
        blur(temp, dst, Size(i, i), Point(-1, -1));
    }
    
    for( int y = 0; y < input.rows; y++ ) {
        for( int x = 0; x < input.cols; x++ ) {
            for( int c = 0; c < input.channels(); c++ ) {
                if (y > target_y && x > target_x && y < (target_height + target_y) && x < (target_length + target_x)) {
                    dst.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( input.at<Vec3b>(y,x)[c] );
                }
                dst.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( (input.at<Vec3b>(y,x)[c] + dst.at<Vec3b>(y,x)[c]) / 2);
            }
        }
    }
    
    return dst;
}

//////////////////
// color functions
//////////////////

Mat equalizeIntensity(const Mat& input)
{
    // guard for images that are not RGB   
    if(input.channels() >= 3)
    {
        // declare new Mat in YCrCb coordinates
        Mat ycrcb;
        
        // convert input to YCrCb image
        cvtColor(input, ycrcb, COLOR_BGR2YCrCb);
        
        // split YCrCb image's channels
        vector<Mat> channels;
        split(ycrcb, channels);
        
        // equalize histogram of intensity photo
        equalizeHist(channels[0], channels[0]);
        
        // merge channels back into YCrCb image and convert back to RGB
        Mat result;
        merge(channels, ycrcb);
        cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
        
        return result;
    }
    
    return Mat();
}

Mat linearContrast(Mat& input, double alpha, double beta){
    
    if( input.empty() ) {
        cout << "Could not open or find the image!\n" << endl;
        return input;
    }
    
    Mat new_image = Mat::zeros( input.size(), input.type() );
    
    for( int y = 0; y < input.rows; y++ ) {
        for( int x = 0; x < input.cols; x++ ) {
            for( int c = 0; c < input.channels(); c++ ) {
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*input.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }
    
    return new_image;
}

////////////////////////////////////////////////////
// display functions (mostly for debugging purposes)
////////////////////////////////////////////////////

void display1(const Mat& im1, string windowName){
    // show one image
    imshow(windowName, im1);
}

void display2(const Mat& im1, const Mat& im2, string windowName) {
    // acquire image sizes
    Size sz1 = im1.size();
    Size sz2 = im2.size();
    
    // declare new Mat with appropriate dimensions
    Mat im3(sz1.height, sz1.width+sz2.width, CV_8UC3);
    
    // copy image 1 (2) to left (right) pane of image
    Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
    im1.copyTo(left);
    Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
    im2.copyTo(right);
    
    // display both images side-by-side
    imshow(windowName, im3);
}
