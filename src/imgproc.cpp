#include "imgproc.hpp"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/////////////////
// blur functions
/////////////////

void applyBoxBlur(Mat& input, const int MAX_KERNEL_LENGTH, detections locations){
    // input guard
    if( input.empty() ) {
        cout << "Image not found.\n" << endl;
        return;
    }
    
    Mat dst, temp;
    
    temp = input.clone();
    
    // black out detected objects
    for (int i = 0; i < locations.classIDs.size(); i++){
        removeException(temp, MAX_KERNEL_LENGTH, locations.x1[i], locations.y1[i], locations.x2[i], locations.y2[i]);
    }
    
    // blur images with blacked out objects
    for( int i = 1 ; i < MAX_KERNEL_LENGTH ; i+=2) {
        blur(temp, dst, Size(i, i), Point(-1, -1));
    }
    
    // return blacked out boxed to original values
    for (int i = 0; i < locations.classIDs.size(); i++){
        addException(dst, input, locations.x1[i], locations.y1[i], locations.x2[i], locations.y2[i]);
    }
    
    input = dst;
}

void removeException(Mat& input, const int MAX_KERNEL_LENGTH, int x1, int y1, int x2, int y2){
    // input guard
    if( input.empty() ) {
        cout << "Image not found.\n" << endl;
        return;
    }
    
    // check for out of bounds exceptions
    
    
    
    // remove exceptions
    
    for( int y = y1 + MAX_KERNEL_LENGTH; y < (y2 - MAX_KERNEL_LENGTH); y++ ) {
        for( int x = x1 + MAX_KERNEL_LENGTH; x < (x2 - MAX_KERNEL_LENGTH); x++ ) {
            for( int c = 0; c < input.channels(); c++ ) {
                input.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(0);
            }
        }
    }
}

void addException(Mat& input, Mat& original, int x1, int y1, int x2, int y2){
    // input guard
    if( input.empty() ) {
        cout << "Image not found.\n" << endl;
        return;
    }
    
    // check for out of bounds exceptions
    
    
    
    // add exceptions
    
    for( int y = y1; y < y2; y++ ) {
        for( int x = x1; x < x2; x++ ) {
            for( int c = 0; c < input.channels(); c++ ) {
                input.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(original.at<Vec3b>(y,x)[c]);
            }
        }
    }
}

//////////////////
// color functions
//////////////////

void equalizeIntensity(Mat& input)
{
    // input guard
    if( input.empty() ) {
        cout << "Image not found.\n" << endl;
        return;
    }
    
    // guard for images that are not RGB
    if(input.channels() >= 3)
    {
        // declare new Mat in YCrCb coordinates
        Mat temp;
        
        // convert input to YCrCb image
        cvtColor(input, temp, COLOR_BGR2YCrCb);
        
        // split YCrCb image's channels
        vector<Mat> channels;
        split(temp, channels);
        
        // equalize histogram of intensity photo
        equalizeHist(channels[0], channels[0]);
        
        // merge channels back into YCrCb image and convert back to RGB
        merge(channels, temp);
        cvtColor(temp, input, COLOR_YCrCb2BGR);
    }
}

void linearContrast(Mat& input, double alpha, double beta){
    // input guard
    if( input.empty() ) {
        cout << "Image not found.\n" << endl;
        return;
    }
    
    // linear contrast adjustment
    for( int y = 0; y < input.rows; y++ ) {
        for( int x = 0; x < input.cols; x++ ) {
            for( int c = 0; c < input.channels(); c++ ) {
                input.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*input.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }
}

////////////////////////////////////////////////////
// display functions (mostly for debugging purposes)
////////////////////////////////////////////////////

void display1(const Mat& im1, string windowName){
    // input guard
    if( im1.empty() ) {
        cout << "Image not found.\n" << endl;
        return;
    }
    
    // show one image
    imshow(windowName, im1);
}

void display2(const Mat& im1, const Mat& im2, string windowName) {
    // input guards
    if(im1.empty() || im2.empty()) {
        cout << "Image not found.\n" << endl;
        return;
    }
    
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
