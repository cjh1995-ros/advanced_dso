#include "Util/ImageReader.hpp"

using namespace adso;

int main()
{
    ImageReader image_reader("data/sequence_09/images", cv::Size(640, 480));
    cv::Mat img = image_reader.readImage(0);
    std::cout << "Image size: " << img.size() << std::endl;
    return 0;
}