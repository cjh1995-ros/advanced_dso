#include "Util/ImageReader.hpp"
#include <gtest/gtest.h>

namespace adso{

TEST(IMAGE_READER_TEST, INITALIZE_TEST)
{
    ImageReader image_reader("data/sequence_09/images");
    cv::Mat img = image_reader.readImage(0);
    ASSERT_EQ(img.size(), cv::Size(640, 480));
}

}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}