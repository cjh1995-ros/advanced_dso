#include "vignette_model.hpp"
#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace adso 
{

TEST(TestVignetteOperate, TestInitVig)
{
    cv::Mat image = (cv::Mat_<uchar>(4, 4) << 0, 0, 0, 0,
                                              0, 0, 0, 0,
                                              0, 0, 0, 0,
                                              0, 0, 0, 0);
    cv::Size img_size = image.size();
    cv::Point2d center = {1.5, 1.5};

    double v1 = 1.0;
    double v2 = 1.0;
    double v3 = 1.0;

    VignetteModel vig(v1, v2, v3, img_size, center);

    // Check vignette factor is inited well
    EXPECT_EQ(vig.v1(), v1);
    EXPECT_EQ(vig.v2(), v2);
    EXPECT_EQ(vig.v3(), v3);

    // Check image size is inited well
    EXPECT_EQ(vig.cvsize().width, img_size.width);
    EXPECT_EQ(vig.cvsize().height, img_size.height);

    // Check center value is inited well
    EXPECT_EQ(vig.center().x, center.x);
    EXPECT_EQ(vig.center().y, center.y);

    // Check max radius is inited well
    EXPECT_DOUBLE_EQ(vig.GetMaxRadius(), 2.121320343559643);
}

TEST(TestVignetteOperate, TestGetNormedRadius)
{
    cv::Mat image = (cv::Mat_<uchar>(4, 4) << 0, 0, 0, 0,
                                              0, 0, 0, 0,
                                              0, 0, 0, 0,
                                              0, 0, 0, 0);
    cv::Size img_size = image.size();
    cv::Point2d center = {1.5, 1.5}; 
    cv::Point2d px1 = {1.0, 1.5}; 
    cv::Point2d px2 = {3.0, 3.0}; // normed radius = 1.0
    double v1 = 1.0;
    double v2 = 1.0;
    double v3 = 1.0;

    VignetteModel vig(v1, v2, v3, img_size, center);

    // Check normed radius is calculated well
    EXPECT_DOUBLE_EQ(vig.GetNormedRadius(center), 0.0);

    // Check normed radius is calculated well
    EXPECT_DOUBLE_EQ(vig.GetNormedRadius(px1), 0.2357022603955158);

    // Check normed radius is calculated well
    EXPECT_DOUBLE_EQ(vig.GetNormedRadius(px2), 1.0);
}

TEST(TestVignetteOperate, TestGetVignetteFactor)
{
    cv::Mat image = (cv::Mat_<uchar>(4, 4) << 0, 0, 0, 0,
                                              0, 0, 0, 0,
                                              0, 0, 0, 0,
                                              0, 0, 0, 0);
    cv::Size img_size = image.size();
    cv::Point2d center = {1.5, 1.5}; 
    cv::Point2d px1 = {1.0, 1.5}; 
    cv::Point2d px2 = {3.0, 3.0}; // normed radius = 1.0
    double v1 = 1.0;
    double v2 = 1.0;
    double v3 = 1.0;

    VignetteModel vig(v1, v2, v3, img_size, center);

    // Check vignette factor is calculated well
    EXPECT_DOUBLE_EQ(vig.GetVignetteFactor(center), 1.0);
    EXPECT_DOUBLE_EQ(vig.GetVignetteFactor(px1), 1.0588134430727023);
    EXPECT_DOUBLE_EQ(vig.GetVignetteFactor(px2), 4.0);
}

} // namespace adso