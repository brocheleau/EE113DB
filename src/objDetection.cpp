
#include <iostream>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "objDetection.hpp"


using namespace cv;
using namespace dnn;


void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    std::cout << "Detection Output\n";
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    CV_Assert(outs.size() == 1);
    float* data = (float*)outs[0].data;
    for (size_t i = 0; i < outs[0].total(); i += 7)
    {
        float confidence = data[i + 2];
        if (confidence > confThreshold)
        {
            int left = (int)(data[i + 3] * frame.cols);
            int top = (int)(data[i + 4] * frame.rows);
            int right = (int)(data[i + 5] * frame.cols);
            int bottom = (int)(data[i + 6] * frame.rows);
            int width = right - left + 1;
            int height = bottom - top + 1;
            classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
            boxes.push_back(Rect(left, top, width, height));
            confidences.push_back(confidence);
        }
    }
    
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}


void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    
    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

        std::cout<<"Label: " << label << "\nLeft, top: " << left << ", " << top << "\nRight, bottom: "<< right << ", " << bottom << "\n";

    
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}


void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}