#ifndef OBJDETECTION_H
#define OBJDETECTION_H
#include <iostream>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct detections {
    vector<string> classIDs;
    vector<int> x1;
    vector<int> y1;
    vector<int> x2;
    vector<int> y2;
};

static float confThreshold, nmsThreshold;
static std::vector<std::string> classes = {
//    "Apple",
//    "Bagel",
//    "Banana",
//    "Computer Keyboard",
//    "Computer Mouse",
//    "Mug",
//    "Laptop",
//    "Person",
//    "Ruler",
//    "Sunglasses"
//};
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"",
"backpack",
"umbrella",
"",
"",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"",
"dining table",
"",
"",
"toilet",
"",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
}; 
/*
*/; 

detections postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);

std::vector<String> getOutputsNames(const Net& net);

#endif
