#include "select.hpp"
#include "util/pixel_operate.hpp"
#include "util/logging.hpp"

namespace adso
{

int Proj2Mask(const DepthPointGrid& points1,
              cv::Mat& mask,
              double scale,
              int dilate)
{
    CHECK_GT(scale, 0);
    CHECK_LE(scale, 1);
    CHECK_GE(dilate, 0);

    int n_pixels = 0;  // number of masked out points
    for (const auto& point : points1)
    {
        if (!point.InfoOk()) continue;

        // scale to mask level, because grid is in full res
        const auto px_s = ScalePix(point.px(), scale);
        const auto px_i = RoundPix(px_s);
        // skip if oob
        if (IsPixOut(mask, px_i, dilate)) continue;

        // update mask, return bool value. 그리고 val을 255만으로 채움
        n_pixels += MatSetWin<uchar>(mask, px_i, {dilate, dilate}, 255);
    }
}

PixelGrad FindMaxGrad(const cv::Mat& image,
                      const cv::Rect& win,
                      const cv::Mat& mask = cv::Mat(),
                      int max_grad = 128) noexcept
{
    PixelGrad pxg{};
    for (int wr = 0; wr < win.height; ++wr)
    {
        for (int wc = 0; wc < win.width; ++wc)
        {
            const cv::Point2i px{wc + win.x, wr + win.y};

            // check if px in mask is occupied, if yes then skip
            if (!mask.empty() && mask.at<uchar>(px) > 0) continue;

            const auto grad = GradAtI<uchar>(image, px);
            const auto grad2 = PointSqNorm(grad);

            // Skip if grad is not bigger than saved
            if (grad2 < pxg.grad2) continue;

            // otherwise save max grad
            pxg.px = px;
            pxg.grad2 = grad2;

            // if grad is big enough in either direction then we can stop early
            if (std::abs(grad.x) >= max_grad || std::abs(grad.y) >= max_grad) return pxg;
        }
    }
}


void CalcPixelGrads(const cv::Mat& image,
                    const cv::Mat& mask,
                    PixelGradGrid& pxgrads,
                    int max_grad,
                    int border = 1,
                    int gsize = 0)
{
    if (!mask.empty())
    {
        CHECK_EQ(mask.type(), CV_8UC1);
        CHECK_EQ(image.rows, mask.rows);
        CHECK_EQ(image.cols, mask.cols);
    }
    CHECK_GE(border, 0);
    CHECK(!pxgrads.empty());
    CHECK(!image.empty());
    CHECK_EQ(image.type(), CV_8UC1);

    const int cell_rows = image.rows / pxgrads.rows();
    const int cell_cols = image.cols / pxgrads.cols();

    
}

} // namespace adso