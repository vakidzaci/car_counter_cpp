#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "nms.cpp"
#include "centroidtracker.h"

#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;
int main(int, char**)
{

    Mat frame;
    Mat bgr;
    Mat grayImage;
    Mat first_frame;
    Mat frameDelta;
    Mat outImg;
    Mat HM;
    Mat gray_blur;
    Mat thresh;
    Mat cropped_image;
    Mat original_img;
    auto centroidTracker = new CentroidTracker(5);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<vector<float>> old_rects;
    RNG rng(12345);
    float x,w,y,h;
    float thresholdFloat = 0.5;
    int delay_counter=0;
    int FRAMES_TO_PERSIST = 3;
    int MIN_SIZE_FOR_MOVEMENT = 2000;
    VideoCapture cap;
    vector<string> IDS;
    cap.open("highway.mp4");
    // check if we succeeded
    cap.read(first_frame);
    int W = first_frame.size().width;
    int H = first_frame.size().height;

    cv::cvtColor(first_frame, first_frame, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(first_frame, first_frame, Size(35, 35), 0);
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        
        original_img = frame.clone();
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayImage, gray_blur, Size(35, 35), 0);
        delay_counter+=1;
        if(delay_counter>=FRAMES_TO_PERSIST){
            delay_counter=0;
            first_frame=gray_blur.clone();            
        }

        cv::absdiff(first_frame, gray_blur, frameDelta);
        cv::threshold(frameDelta, thresh, 5, 255, cv::THRESH_BINARY);
        cv::dilate(thresh, thresh, 5);
        
        cv::findContours( thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
        vector<Rect> boundRect( contours.size() );
        Scalar color = Scalar( 0, 255, 0 );
        vector<vector<float>> vec; 
        for( size_t i = 0; i < contours.size(); i++ )
        {
            if(cv::contourArea(contours[i]) > MIN_SIZE_FOR_MOVEMENT){
                boundRect[i] = boundingRect(contours[i]);
                x = boundRect[i].x;
                w = boundRect[i].width;
                y = boundRect[i].y;
                h = boundRect[i].height;
                vector<float> one_vec = {x,y,x+w,y+h};
                vec.push_back(one_vec);
            }
        }


        vector<vector<int>> rects;
        vector<vector<float>> all_rects;
        for(vector<float> rec: old_rects){
            all_rects.insert(all_rects.end(),rec);
        }
        for(vector<float> rec: vec){
            all_rects.insert(all_rects.end(),rec);
        }
        old_rects = vec;
        std::vector<cv::Rect> reducedRectangle = nms(vec, 0.1);

        for(cv::Rect v : reducedRectangle){
            x = v.x;
            w = v.width;
            y = v.y;
            h = v.height;
            rects.insert(rects.end(), {x, y, x+w, y+h});
        }
        vector<pair<int, pair<int, int>>>  objects = centroidTracker->update(rects);

        if (!objects.empty()) {
            for (auto obj: objects) {
                // cout << obj.second.first << " " << obj.second.second << endl;
                circle(frame, Point(obj.second.first, obj.second.second), 4, Scalar(255, 0, 0), -1);
                string ID = std::to_string(obj.first);
                cv::putText(frame, ID, Point(obj.second.first - 10, obj.second.second - 10),
                            FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0), 2);
                if (std::find(IDS.begin(), IDS.end(), ID) == IDS.end())
                {
                    IDS.insert(IDS.end(),ID);
                    cropped_image = original_img(Range(std::max(obj.second.second-100,0),min(H-1,obj.second.second+100)), Range(std::max(0,obj.second.first-100), min(W-1,obj.second.first+100)));
                    cv::imwrite("saved_cars/"+ID+".jpg", cropped_image);

                }

            }

        }


        for(cv::Rect rect: reducedRectangle){
            cv::rectangle(frame, rect,  color,2);
        }
        cv::vconcat(grayImage, thresh, HM);
        cv::resize(HM, outImg, cv::Size(), 0.2, 0.2);
    
        imshow("Live", outImg);
        if (waitKey(5) >= 0)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
