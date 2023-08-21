#include "select.hpp"
#include "util/pixel_operate.hpp"
#include "util/logging.hpp"
#include "util/tbb.hpp"

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
                      const cv::Mat& mask,
                      int max_grad) noexcept
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
                    int border,
                    int gsize)
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

    ParallelFor(
        {border, pxgrads.rows() - border, gsize},
        [&] (int gr)
        {
            for(int gc = border; gc < pxgrads.cols() - border; ++gc)
            {
                auto& pxg = pxgrads.at(gr, gc);

                const cv::Rect win = {gc * cell_cols + 1, gr * cell_rows + 1,
                                      cell_cols - 1, cell_rows - 1};

                pxg = FindMaxGrad(image, win, mask, max_grad);                                      
            }
        }
    );

}


int PixelSelector::Select(const ImagePyramid& grays, int gsize)
{
    // Make sure pyramid has enough levels
    CHECK_GT(grays.size(), cfg_.set_vel);

    // Make sure cell size is large enough for top of pyramid
    const auto cell_too_small = cfg_.cell_size < std::pow(2, grays.size() - 1);
    VLOG_IF(1, cell_too_small) << "Cell size is too small for top of pyramid";

    // Allocate storage if needed
    Allocate(grays);
    pixels_.reset(cv::Point{-1, -1});
    pxgrads_.reset();

    CalcPixelGrads(grays[cfg_.set_vel], occ_mask_, pxgrads_, cfg_.max_grad, grid_border_, gsize);

    int n_pixels = 0;
    const auto gray_top = grays.at(0);
    const auto upscale = static_cast<int> (std::pow(2, cfg_.set_vel));

    // Do a first pass of selection using the current min_grad
    const auto n1 = SelectPixels(gray_top, upscale, cfg_.min_grad, gsize);
    n_pixels += n1;
    // Based on the number of pixels, determin how we should change min_grad
    const double ratio1 = static_cast<double> (n_pixels) / pixels_.area();

    // Do a second pass of selection using the new min_grad
    if (cfg_.reselect && ratio1 < cfg_.min_ratio)
        n_pixels += SelectPixels(gray_top, upscale, cfg_.min_grad * 2, gsize);

    const auto ratio2 = static_cast<double> (n_pixels) / pixels_.area();
    const auto new_min_grad = std::clamp(AdaptMinGrad(ratio1, ratio2), 2, 32);

    // Log ~~~

    cfg_.min_grad = new_min_grad;

    return n_pixels;    
}

int PixelSelector::SelectPixels(const cv::Mat& gray,
                                int upscale,
                                int min_grad,
                                int gsize)
{
    if (upscale == 1) return SelectPixels(min_grad);

    const int min_grad2 = min_grad * min_grad;
    return ParallelReduce(
        {0, pixels_.rows(), gsize},
        0,
        [&](int gr, int& n)
        {
            for (int gc = 0; gc < pixels_.cols(); ++gc)
            {
                auto& px = pixels_.at(gr, gc);
                if (!IsPixBad(px)) continue;

                const auto& pxg = pxgrads_.at(gr, gc);
                if (pxg.grad2 < min_grad2) continue;

                // Find max grad pixel within this small window
                const cv::Rect win = {pxg.px.x * upscale, pxg.px.y * upscale,
                                      upscale, upscale};

                px = FindMaxGrad(gray, win).px;
                ++n;
            }
        },
        std::plus<>{}
    );
}

int PixelSelector::SelectPixels(int min_grad)
{
    int n_pixels = 0;

    const int min_grad2 = min_grad * min_grad;

    for (int i=0; i < pixels_.area(); ++i)
    {
        // Skip if already selected
        auto& px = pixels_.at(i);
        if (!IsPixBad(px)) continue;

        // Skip with too small gradients
        const auto& pxg = pxgrads_.at(i);
        if (pxg.grad2 < min_grad2) continue;

        px = pxg.px;
        ++n_pixels;
    }
}

int PixelSelector::SetOccMask(absl::Span<const DepthPointGrid> points1s)
{
    CHECK(!occ_mask_.empty());
    occ_mask_.setTo(0);

    const auto scale = std::pow(2, -cfg_.set_vel);
    int n_pixels = 0;

    for (const auto& points1 : points1s)
    {
        n_pixels += Proj2Mask(points1, occ_mask_, scale, cfg_.nms_size);
    }
    return n_pixels;
}

int PixelSelector::AdaptMinGrad(double ratio1, double ratio2) const noexcept
{
    // If too many pixels selected, we slightly increase min_grad to detect fewer
    if (ratio1 > cfg_.max_ratio) return cfg_.min_grad + 2;

    // If not enough in 1st round
    if (ratio1 < cfg_.min_ratio)
    {
        // If still not enough in 2nd round, just half min_grad
        if (ratio2 < cfg_.max_ratio) return cfg_.min_grad / 2;
        // Only decrease by a bit if got enough in 2nd round
        return cfg_.min_grad - 2;
    }

    // Otherwise, don't change
    return cfg_.min_grad;
}


size_t PixelSelector::Allocate(const cv::Size& top_size,
                               const cv::Size& sel_size)
{
    // Allocate grid
    const cv::Size grid_size{top_size.width / cfg_.cell_size,
                             top_size.height / cfg_.cell_size};

    if (pixels_.empty())
    {
        pixels_.resize(grid_size, {-1, -1});
        pxgrads_.resize(grid_size);
    }
    else
    {
        CHECK_EQ(pixels_.rows(), grid_size.height);
        CHECK_EQ(pixels_.cols(), grid_size.width);
    }

    // Allocate mask
    if (occ_mask_.empty())
        occ_mask_ = cv::Mat::zeros(sel_size, CV_8UC1);
    else
    {
        CHECK_EQ(occ_mask_.rows, sel_size.height);
        CHECK_EQ(occ_mask_.cols, sel_size.width);
    }

    return occ_mask_.total() * occ_mask_.elemSize() +
           pixels_.size() * sizeof(cv::Point2d) +
           pxgrads_.size() * sizeof(PixelGrad);
}

size_t PixelSelector::Allocate(const ImagePyramid& grays)
{
    const auto top_size = grays.at(0).size();
    const auto sel_size = grays.at(cfg_.set_vel).size();
    return Allocate(top_size, sel_size);
}
} // namespace adso