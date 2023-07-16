#include "Util/ImageReader.hpp"
#include "Image/Frame.hpp"
#include <gtest/gtest.h>

using namespace adso;

TEST(IMAGE_READER_TEST, INITALIZE_TEST)
{
    std::string image_folder = "data/dataset-corridor4_512_16/dso/cam0/images";
    int index = 100;
    ImageReader image_reader(image_folder);
    cv::Mat img = image_reader.at(index);
    ASSERT_EQ(img.size(), cv::Size(512, 512));
}

TEST(STEREO_IMAGE_READER_TEST, READ_IMAGES)
{
    std::string image_folder = "data/dataset-corridor4_512_16/dso";

    StereoImageReader stereo_image_reader(image_folder);

    ASSERT_TRUE(stereo_image_reader.isSameSize());
}

TEST(STEREO_FRAME_TEST, READ_IMAGES)
{
    std::string image_folder = "data/dataset-corridor4_512_16/dso";
    size_t idx = 100;

    StereoImageReader stereo_image_reader(image_folder);

    std::vector<cv::Mat> frames = stereo_image_reader.at(idx);

    StereoFrame stereo_frame(frames, idx);

    for (int i = 0; i < 4; i++)
        ASSERT_TRUE(stereo_frame.isSameSize(i));
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}