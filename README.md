# EE113DB
Project folder for EE113D- Signal Digital Processing Design, UCLA Fall 2018/Winter 2019

Team Members: Blake Rocheleau, Julia Schell, Peter Slaats

Project Description:

 The overall goal of this project is to process video data real-time and maintain focus on a specified object as the camera moves (meaning the object moves around in the frame). A webcam would capture the video and send the data to the computer via USB. The object of focus is specified at the start of recording. We will construct and train a CNN to perform real-time object detection to detect the target object in each frame. We will then apply image processing techniques to simulate auto-focus on the image relative to its surroundings. The target object will be enhanced with some combination of deblurring, denoising, adjusting contrast, and/or sharpening edges. Conversely, the image will be ‘de-focused’ proportional to distance from the object in focus by blurring, adding noise, and/or adjusting the histogram. 
 
 Arguments for running main.cpp:
 --aliasName
 --thr=<confidence threshold>
 --input=<input image>
 --config=<Path/to/ssd_mobilenet_v2_coco.pbtxt>
 --model=<Path/to/frozen_inference_graph.pb>
 
 Environment Variables:
 OPENCV_DNN_TEST_DATA_PATH = <Path/to/testdata>

Example: 

To use the pre-trained model (definitely working):
./autoFocus \
--input=sample_images/apple_desk.jpg \
--config=trained_model.pb/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.pbtxt \
--model=trained_model.pb/pre-trained/frozen_inference_graph.pb \
--subject="mouse"

To use my model (not currently working):
./autoFocus \
--input=sample_images/apple_desk.jpg \
--config=trained_model.pb/gsc_v1/ssd_mobilenet_v2_coco.pbtxt \
--model=trained_model.pb/gsc_v1/frozen_inference_graph.pb

To compile: 
clang++ $(pkg-config --cflags --libs /usr/local/Cellar/opencv/4.0.1/lib/pkgconfig/opencv4.pc) -Iinclude -std=c++11 src/autoFocus.cpp src/objDetection.cpp src/imgproc.cpp -o autoFocus

