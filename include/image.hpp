#pragma once

#include <opencv2/core/mat.hpp>


namespace adso
{

using ImagePyramid = std::vector<cv::Mat>;

/// @brief 
template <typename T>
bool MatSetRoi(cv::Mat& mat, 
               cv::Rect roi, 
               const T& val) noexcept
{
    // 이미지와 roi의 교차 영역을 계산
    roi &= cv::Rect{0, 0, mat.cols, mat.rows};
    if (roi.empty()) return false;

    // mat의 roi 영역을 val로 채움
    for (int r = 0; r < roi.height; ++r) 
    {
        for (int c = 0; c < roi.width; ++c)
        {
            mat.at<T>(roi.y + r, roi.x + c) = val;
        }
    }

    return true;
}

/// @brief Set window in image to val, given center and half size
///        윈도우 사이즈 = half size * 2 + 1
///        윈도우는 중심점 px 를 기준으로 fullsize 를 갖고 val로 채워짐
template <typename T>
bool MatSetWin(cv::Mat& mat,
               const cv::Point& px,
               const cv::Point& half_size,
               const T& val) noexcept
{
    const cv::Size full_size{half_size.x * 2 + 1, half_size.y * 2 + 1};
    const cv::Rect roi{px - half_size, full_size};
    return MatSetRoi(mat, roi, val);
}

/// @brief  Threshold depth by max depth
void ThresholdDepth(const cv::Mat& depth, cv::Mat& depth_out, double max_depth);

/// @brief Crop image (top-left) to size that can be evenly divided by factor
///        This crop should not change the camera intrinsics
///        원본 이미지를 factor로 가장 근접하게 cropping 하는 방법. 예 10x10, 3 -> 9x9
cv::Mat CropImageFactor(const cv::Mat& image, int factor);

/// @brief Generate a random image gray scale image [0, 255]
cv::Mat MakeRandMat8U(int rows, int cols = 0);

/// @brief Check if given image pyramid is valid
bool IsImagePyramid(const ImagePyramid& images);

/// @brief Compute number of bytes in image pyramid
size_t GetTotalBytes(const ImagePyramid& images);

/// @brief Check if given image pyramids are stereo
bool IsStereoPair(const ImagePyramid& images0, const ImagePyramid& images1);

/// @brief Construct an image pyramid
void MakeImagePyramid(const cv::Mat& image, int levels, ImagePyramid& pyramid);

/// @brief Make a gradient image for visulization (stores gradient magnitude)
void MakeGradImage(const cv::Mat& image, cv::Mat& grad);

/// @brief Copy image pyramid from source to target
void CopyImagePyramid(const ImagePyramid& source, ImagePyramid& target);

/// @brief Make a gradient image pyramid
void MakeGradPyramid(const ImagePyramid& images,
                     ImagePyramid& pyramid,
                     bool to_uint8 = false);

} // namespace adso