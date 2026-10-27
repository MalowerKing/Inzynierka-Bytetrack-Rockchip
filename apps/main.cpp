#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>   // VideoCapture, imwrite (no videoio/imgcodecs in opencv-mobile)

int main()
{
    cv::Mat frame;
    cv::VideoCapture cap;

    if (!cap.open(0)) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    std::cout << "Start grabbing" << std::endl;
    for (int i = 0; i < 100; ++i)
    {
        cap >> frame;                       // if this doesn't compile, use whatever the grep above shows
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // TODO: RKNN inference + ByteTrack update on `frame`
    }

    cv::imwrite("/tmp/frame.jpg", frame);   // check the result by copying the file off the board
    cap.release();
    return 0;
}
