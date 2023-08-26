#pragma once
#include "point.hpp"
#include <opencv2/imgproc.hpp>

namespace adso
{

void DrawSelectedPixels(cv::Mat img,
                        const PixelGrid& pixels,
                        const cv::Scalar& color,
                        int dilate)
{
    for (const auto& px : pixels)
    {
        if (IsPixBad(px)) continue;
        cv::circle(img, px, 1, color, -1);
    }
}


} // namespace adso