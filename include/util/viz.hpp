#pragma once
#include "point.hpp"
#include <opencv2/imgproc.hpp>

namespace adso
{

void DrawRectangle(cv::Mat& img,
                   const cv::Point2i& pixel,
                   const cv::Scalar& color,
                   int dilate)
{
    cv::Rect rect{pixel.x - dilate, pixel.y - dilate, 2*dilate, 2*dilate};
    cv::rectangle(img, rect, color, 1);
}

void DrawSelectedPixels(cv::Mat img,
                        const PixelGrid& pixels,
                        const cv::Scalar& color,
                        int dilate)
{
    for (const auto& px : pixels)
    {
        if (IsPixBad(px)) continue;
        DrawRectangle(img, px, color, dilate);
    }
}


} // namespace adso