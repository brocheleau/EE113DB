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

    "Person",
    "Bike",
    "Car",
    "Motorcycle",
    "Airplane",
    "Bus",
    "Train",
    "Truck",
    "Boat",
    "Traffic Light",
    "Fire Hydrant",
    "",
    "Stop sign",
    "Meter",
    "Bench",
    "Bird",
    "Cat",
    "Dog",
    "Horse",
    "Sheep",
    "Cow",
    "Elephant",
    "Bear",
    "Zebra",
    "Giraffe",
    "",
    "Backpack",
    "Umbrella",
    "",
    "",
    "Handbag",
    "Tie",
    "Suitcase",
    "Frisbee",
    "Skiis",
    "Snowboard",
    "Sports Ball",
    "Kite",
    "Baseball Bat",
    "Baseball Glove",
    "Skateboard",
    "Surfboard",
    "Tennis Racket",
    "Bottle",
    "",
    "Wine Glass",
    "Cup",
    "Fork",
    "Knife",
    "Spoon",
    "Bowl",
    "Banana",
    "Apple",
    "Sandwich",
    "Orange",
    "Broccoli",
    "Carrot",
    "Hot Dog",
    "Pizza",
    "Donut",
    "Cake",
    "Chair",
    "Couch",
    "Potted Plant",
    "Bed",
    "",
    "Dining Table",
    "",
    "",
    "Toilet",
    "",
    "TV",
    "Laptop",
    "Mouse",
    "Remote",
    "Keyboard",
    "Cell phone",
    "Microwave",
    "Oven",
    "Toaster",
    "Sink",
    "Fridge",
    "",
    "Book",
    "Clock",
    "Vase",
    "Scissors",
    "Teddy Bear",
    "Hair Drier",
    "Toothbrush"
}; 
/*
*/; 

detections postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);

std::vector<String> getOutputsNames(const Net& net);

#endif