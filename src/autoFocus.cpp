//
//  main.cpp
//  Capstone
//
//  Created by Julia Schell on 1/17/19.
//  Copyright © 2019 Julia Schell. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"
#include "objDetection.hpp"
#include "imgproc.hpp"

string keys =
"{ help  h     | | Print help message. }"
"{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
"{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
"{ device      |  0 | camera device number. }"
"{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
"{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
"{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
"{ thr         | .5 | Confidence threshold. }"
"{ nms         | .4 | Non-maximum suppression threshold. }"
"{ subject s   | | Subject class to focus on. }"
"{ backend     |  0 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation }"
"{ target      | 0 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU }";


using namespace cv;
using namespace dnn;
using namespace std;

string subject = "mouse";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    
    const string modelName = parser.get<String>("@alias");
    const string zooFile = parser.get<String>("zoo");
    
    keys += genPreprocArguments(modelName, zooFile);
    
    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run object detection deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    confThreshold = parser.get<float>("thr");
    nmsThreshold = parser.get<float>("nms");
    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    string modelFilePath = parser.get<String>("model");
    string configFilePath = parser.get<String>("config");
    CV_Assert(parser.has("model"));
    string modelPath = findFile(parser.get<String>("model"));
    string configPath = findFile(parser.get<String>("config"));
    CV_Assert(parser.has("subject"));
    //string subject = parser.get<string>("subject");
    
    // Open file with classes names.
    if (parser.has("classes"))
    {
        string file = parser.get<String>("classes");
        ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        string line;
        while (getline(ifs, line))
        {
            classes.push_back(line);
        }
    }
    
    // Load a model.
    Net net = readNet(modelPath, configPath, parser.get<String>("framework"));
    net.setPreferableBackend(parser.get<int>("backend"));
    net.setPreferableTarget(parser.get<int>("target"));
    vector<String> outNames = net.getUnconnectedOutLayersNames();
    
    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    int initialConf = (int)(confThreshold * 100);
    createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);
    
    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(parser.get<int>("device"));
    
    if (!cap.isOpened())
    {
        cout  << "Could not open the input video." << endl;
        return -1;
    }

    
    Size S = Size((int) cap.get(CAP_PROP_FRAME_WIDTH), (int) cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter outVideo("demo_video_samples.avi", VideoWriter::fourcc('M','J','P','G'), cap.get(CAP_PROP_FPS)-10, S);
    
    cout << "Starting rendering." << endl;

    Mat frame;
    // collect video frames
    for (int i = 0; i < 100 ; i++) {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }
        waitKey(1);
        outVideo.write(frame);
        imshow(kWinName, frame);
    }
    
    cout << "Collection complete." << endl;
    
    outVideo.release();
    cap.release();
    
    VideoCapture newCap("demo_video_samples.avi");
    
    if (!newCap.isOpened())
    {
        cout  << "Could not open the input video." << endl;
        return -1;
    }
    
    S = Size((int) newCap.get(CAP_PROP_FRAME_WIDTH), (int) newCap.get(CAP_PROP_FRAME_HEIGHT));
    
    VideoWriter finalVideo("demo_video_output.avi", VideoWriter::fourcc('M','J','P','G'), newCap.get(CAP_PROP_FPS), S);
    
    // Process frames.
    Mat blob;
    while (waitKey(1) < 0)
    {
        newCap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        // Create a 4D blob from a frame.
        Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
                     inpHeight > 0 ? inpHeight : frame.rows);
        blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);

        // Run a model.
        net.setInput(blob);
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
        {
            resize(frame, frame, inpSize);
            Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
            net.setInput(imInfo, "im_info");
        }
        vector<Mat> outs;
        net.forward(outs, outNames);

        detections results = postprocess(frame, outs, net);

        // intensity histogram equalization
        equalizeIntensity(frame);

        // check if desired object is in image
        detections desiredObjects;
        for (unsigned int i = 0; i<results.classIDs.size(); i++){
            if (results.classIDs[i] == subject){
                desiredObjects.classIDs.push_back(subject);
                desiredObjects.x1.push_back(results.x1[i]);
                desiredObjects.y1.push_back(results.y1[i]);
                desiredObjects.x2.push_back(results.x2[i]);
                desiredObjects.y2.push_back(results.y2[i]);
            }
        }

        // if desired object appears, add to blur exceptions list
        if (desiredObjects.classIDs.size() > 0) {
            applyBoxBlur(frame, 7, desiredObjects);
        }
        // else, use every detected object on the blur exceptions list
        else {
            applyBoxBlur(frame, 7, results);
        }
        
        finalVideo.write(frame);

//        // Put efficiency information.
//        vector<double> layersTimes;
//        double freq = getTickFrequency() / 1000;
//        double t = net.getPerfProfile(layersTimes) / freq;
//        string label = format("Inference time: %.2f ms", t);
//        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    }
    
    cout << "Finished rendering." << endl;
    
    return 0;
}
