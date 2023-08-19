#include "image.hpp"
#include <opencv2/imgproc.hpp>
#include "util/logging.h"


namespace adso
{

void ThresholdDepth(const cv::Mat& depth, cv::Mat& depth_out, double max_depth)
{
    cv::threshold(depth, depth_out, max_depth, 0, cv::THRESH_TOZERO_INV);
}

cv::Mat CropImageFactor(const cv::Mat& image, int factor)
{
    CHECK_GT(factor, 0);
    const int new_rows = (image.rows / factor) * factor;
    const int new_cols = (image.cols / factor) * factor;
    // Do nothing if cropped image is the same size as input image
    if (new_rows == image.rows && new_cols == image.cols) return image;

    cv::Mat cropped;
    cv::Mat(image, cv::Rect{0, 0, new_cols, new_rows}).copyTo(cropped);
    return cropped;
}

cv::Mat MakeRandMat8U(int rows, int cols = 0)
{
    if (cols == 0) cols = rows;
    cv::Mat mat{rows, cols, CV_8UC1};
    cv::randu(mat, cv::Scalar{0}, cv::Scalar{255});
    return mat;
}

bool IsImagePyramid(const ImagePyramid& images)
{
    if (images.empty()) return false;

    for (size_t i=1; i<images.size()l ++i)
    {
        const auto& below = images[i-1];
        const auto& above = images[i];

        // Check that above should be half of below
        if (above.rows != std::ceil(below.rows / 2.0) ||
            above.cols != std::ceil(below.cols / 2.0))
        {
            return false;
        }
    }
    return true;
}

size_t GetTotalBytes(const ImagePyramid& images)
{
    size_t n{0};
    for (const auto& image : images)
    {
        n += image.total() * image.elemSize();
    }
    return n;
}

bool IsStereoPair(const ImagePyramid& images0, const ImagePyramid& images1)
{
    if (images0.size() != images1.size()) return false;

    for (size_t i=0; i<images0.size(); ++i)
    {
        const auto& im0 = images0.at(i);
        const auto& im1 = images1.at(i);
        if (im0.rows != im1.rows || im0.cols != im1.cols)
        {
            return false;
        }
    }
    return true;
}

void MakeImagePyramid(const cv::Mat& image, int levels, ImagePyramid& pyramid)
{
    CHECK(!image.empty());

    pyramid.resize(levels);
    pyramid[0] = image.clone();
    for (int l=1; l<levels; ++l)
    {
        cv::pyrDown(pyramid[l-1], pyramid[l]);
    }

    // now go back and blur the first image (otherwise we will double blur)
    cv::GaussianBlur(pyramid[0], pyramid[0], {3, 3}, 0);
}

void MakeGradImage(const cv::Mat& image, cv::Mat& grad)
{
    cv::Mat gx;
    cv::Mat gy;
    const double scale = 1.0 / 4.0 / 255.0;
    cv::Sobel(image, gx, CV_32FC1, 1, 0, 3, scale);
    cv::Sobel(image, gy, CV_32FC1, 0, 1, 3, scale);
    cv::magnitude(gx, gy, grad);
}

void CopyImagePyramid(const ImagePyramid& source, ImagePyramid& target)
{
    target.resize(source.size());
    for (size_t i=0; i<source.size(); ++i)
    {
        source[i].copyTo(target[i]);
    }
}

void MakeGradPyramid(const ImagePyramid& images,
                     ImagePyramid& grads,
                     bool to_uint8 = false)
{
    grads.resize(images.size());
    for (size_t i=0; i<images.size(); ++i)
    {
        MakeGradImage(images[i], grads[i]);
        if (to_uint8)
        {
            grads[i].convertTo(grads[i], CV_8UC1, 255);
        }
    }
}


} // namespace adso