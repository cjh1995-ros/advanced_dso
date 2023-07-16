#include "Util/ImageReader.hpp"

using namespace adso;

int main()
{
    ImageReader image_reader("data/sequence_09/images", cv::Size(640, 480));
    cv::Mat img = image_reader.readImage(0);
    
    int n_images = image_reader.getNumImages();

    for(int i = 0; i<n_images; i++)
    {
        cv::Mat img = image_reader.readImage(i);
        cv::imshow("img", img);
        cv::waitKey(0);
    }
}