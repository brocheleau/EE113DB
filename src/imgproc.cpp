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
    
    Mat padded_img;
    int padding = MAX_KERNEL_LENGTH;
    padded_img.create(input.rows + 2*padding, input.cols + 2*padding, input.type());
    padded_img.setTo(Scalar::all(0));
    input.copyTo(padded_img(Rect(padding, padding, input.cols, input.rows)));
    
    bool insideBox = false;
    
    for (int y = padding; y < padded_img.rows - padding; y++) {
        for (int x = padding; x < padded_img.cols - padding; x++) {
            // check if x,y is within a bounding box
            for (int i = 0; i < locations.classIDs.size(); i++){
                if ((x-padding) >= locations.x1[i] && (x-padding) < locations.x2[i] && (y-padding) >= locations.y1[i] && (y-padding) < locations.y2[i]) {
                    insideBox = true;
                }
            }
            
            if (insideBox) {
                insideBox = false;
                continue;
            }
            
            int k = MAX_KERNEL_LENGTH;
            double minDistance = 1e10;
            double distance = 0;
            
            // compute distance to box/set kernel size
            for (int i = 0; i < locations.classIDs.size(); i++){
                double xcenter = (locations.x1[i] + locations.x2[i] + 2*padding) / 2;
                double ycenter = (locations.y1[i] + locations.y2[i] + 2*padding) / 2;
                
                double xdistance = (x - xcenter);
                double ydistance = (y - ycenter);
                
                xdistance /= input.cols;
                ydistance /= input.rows;
                
                xdistance = pow(xdistance, 2);
                ydistance = pow(ydistance, 2);
                
                distance = xdistance + ydistance;
                distance = pow(distance, 0.5);
                
                if (distance <= minDistance) {
                    minDistance = distance;
                }
            }
            
            minDistance *= MAX_KERNEL_LENGTH*3;
            k = min(int(minDistance), MAX_KERNEL_LENGTH);
            if (k == 0) { k++; }
            
            // convolve kernel and set pixel value
            int anchor = k / 2;
            double sum = 0;
            
            for (int c = 0; c < padded_img.channels(); c++) {
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        sum += (padded_img.at<Vec3b>((y + j - anchor), (x + i - anchor))[c] / pow(k,2));
                    }
                }
                input.at<Vec3b>(y - padding,x - padding)[c] = saturate_cast<uchar>(int(sum));
                sum = 0;
            }
        }
    }
    //padded_img(Rect(padding, padding, input.cols, input.rows)).copyTo(input);
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
