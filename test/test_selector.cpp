#include "select.hpp"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

namespace adso
{

constexpr int kNumLevels = 2;
constexpr int kImageSize = 640;
constexpr int kCellSize = 16;
const cv::Size kGridSize = {kImageSize / kCellSize, kImageSize / kCellSize};


TEST(TestPixelSelectOperatation, TestFindMaxGrad)
{
    const cv::Mat image = 
        (cv::Mat_<uchar>(4, 4) << 1, 1, 1, 1,
                                  1, 2, 9, 1,
                                  0, 3, 6, 0,
                                  0, 0, 0, 0);
    const cv::Rect win = {1, 1, 2, 2}; // 2x2 window start in (1, 1)

    const auto pxg = FindMaxGrad(image, win);

    EXPECT_EQ(pxg.px.x, 2);
    EXPECT_EQ(pxg.px.y, 2);
    EXPECT_DOUBLE_EQ(pxg.grad2, 22.5); // (9)/2 ^2 + (3-0)/2 ^2
}

TEST(TestPixelSelectFunc, TestAllocate)
{
    PixelSelector selector;
    const cv::Size top_size{160, 320};  // 10 x 20
    const cv::Size sel_size{80, 160};
    EXPECT_EQ(selector.Allocate(top_size, sel_size), 19200);
}

}