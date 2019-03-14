#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// variables for image blurring
int MAX_KERNEL_LENGTH = 31;

int display_dst( int delay );

int main(int argc, const char * argv[]) {
    
    ///////////////////////////////////////////
    //  Video input/output program
    //////////////////////////////////////////
    
    // variable initialization
    Mat frame;
    Mat dst;
    Mat temp;
    VideoCapture cap;            // 0 is the webcam
    
//    double m_alpha = 1.3;           // Simple contrast control
//    int m_beta = 0;                 // Simple brightness control
    
    // try to open camera
    cap.open(0);
    if (!cap.isOpened()) return -1;  // check if there is no video or no way to display it
    
    // create a window and display the video capture
    namedWindow("Video Capture", WINDOW_AUTOSIZE);
    
    for (int i=0; i<1000; i++) {
        cap >> frame;
        frame.copyTo(temp);
        
        // contrast adjustment
        
        int target_length = 400;
        int target_height = 400;

        int target_x = 500, target_y = 50;

        for( int y = target_y + MAX_KERNEL_LENGTH; y < (target_y + target_height - MAX_KERNEL_LENGTH); y++ ) {
            for( int x = target_x + MAX_KERNEL_LENGTH; x < (target_x + target_length - MAX_KERNEL_LENGTH); x++ ) {
                for( int c = 0; c < frame.channels(); c++ ) {
                    temp.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(0);
                }
            }
        }

        for( int i = 1 ; i < MAX_KERNEL_LENGTH ; i+=2) {
            blur(temp, dst, Size(i, i), Point(-1, -1));
        }

        for( int y = 0; y < frame.rows; y++ ) {
            for( int x = 0; x < frame.cols; x++ ) {
                for( int c = 0; c < frame.channels(); c++ ) {
                    if (y > target_y && x > target_x && y < (target_height + target_y) && x < (target_length + target_x)) { continue; }
                    frame.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( (dst.at<Vec3b>(y,x)[c] + frame.at<Vec3b>(y,x)[c]) / 2);
                }
            }
        }
        
        
        
//        for( int y = target_y; y < (target_y + target_height); y++ ) {
//            for( int x = target_x; x < (target_x + target_length); x++ ) {
//                for( int c = 0; c < frame.channels(); c++ ) {
//                    if (y >= frame.rows || x >= frame.cols){ continue; }
//                    frame.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( m_alpha*frame.at<Vec3b>(y,x)[c] + m_beta );
//                }
//            }
//        }
        

        
        imshow("Video Capture", frame);
        waitKey(1);
    }
    return 0;
}

/*

int display_dst( int delay )
{
    imshow( window_name, dst );
    int c = waitKey ( delay );
    if( c >= 0 ) { return -1; }
    return 0;
}

*/

/*
 
 ///////////////////////////////////////////
 //  image contrast adjustment program
 //////////////////////////////////////////
 
 Mat image = imread("/Users/peterslaats/Desktop/ucla.jpg");
 
 if( image.empty() )
 {
 cout << "Could not open or find the image!\n" << endl;
 cout << "Usage: " << argv[0] << " <Input image>" << endl;
 return -1;
 }
 
 Mat new_image = Mat::zeros( image.size(), image.type() );
 
 double m_alpha = 1.0;           // Simple contrast control
 int m_beta = 0;                 // Simple brightness control
 
 cout << " Basic Linear Transforms " << endl;
 cout << "-------------------------" << endl;
 cout << "* Enter the alpha value [1.0-3.0]: "; cin >> m_alpha;
 cout << "* Enter the beta value [0-100]: ";    cin >> m_beta;
 for( int y = 0; y < image.rows; y++ ) {
 for( int x = 0; x < image.cols; x++ ) {
 for( int c = 0; c < image.channels(); c++ ) {
 new_image.at<Vec3b>(y,x)[c] =
 saturate_cast<uchar>( m_alpha*image.at<Vec3b>(y,x)[c] + m_beta );
 }
 }
 }
 imshow("Original Image", image);
 imshow("New Image", new_image);
 waitKey();
 
 
 */


/*
 ///////////////////////////////////////////
 //  histogram equalization program
 //////////////////////////////////////////
 
 Mat src = imread("/Users/peterslaats/Desktop/ucla.jpg", IMREAD_COLOR);
 
 if( src.empty() )
 {
 cout << "Could not open or find the image!\n" << endl;
 cout << "Usage: " << argv[0] << " <Input image>" << endl;
 return -1;
 }
 
 cvtColor(src, src, COLOR_BGR2GRAY );
 Mat dst;
 equalizeHist( src, dst );
 imshow( "Source image", src );
 imshow( "Equalized Image", dst );
 waitKey();
 
 */
