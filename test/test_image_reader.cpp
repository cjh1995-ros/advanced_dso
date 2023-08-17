#include <gtest/gtest.h>
#include <iostream>
#include "image_reader.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace adso {

class ImageReaderTest : public ::testing::Test {
protected:
    // You can do set-up work for each test here.
    ImageReaderTest() {}

    // You can do clean-up work that doesn't throw exceptions here.
    virtual ~ImageReaderTest() {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:
    virtual void SetUp() {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }
};

TEST_F(ImageReaderTest, ReadImageTest) {
    // Assuming you have some images in the "test_images" directory for testing
    ImageReader reader("data/data");

    cv::Mat img = reader.readImage(0); // Read the first image

    std::cout << reader.logging() << std::endl;
    // Check if the image is not empty
    ASSERT_FALSE(img.empty());

    // Check if the image is grayscale
    ASSERT_EQ(img.channels(), 1);

    // You can add more assertions as needed to validate the behavior of your ImageReader class
}

}  // namespace adso

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
