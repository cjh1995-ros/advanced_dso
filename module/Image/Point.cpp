#include "Util/Logging.hpp"

#include "Image/Point.hpp"
#include "Image/Pixel.hpp"

namespace adso
{

void DepthPoint::SetIdepthInfo(double idepth, double info) noexcept
{
    CHECK(PixelOk());
    CHECK_GE(idepth, 0);
    CHECK_LE(info, kMaxInfo);
    idepth_ = idepth;
    info_ = info;
}

std::string DepthPoint::Repr() const
{
    return fmt::format("DepthPoint(uv=({},{}), idepth={:0.4f}, info={})",
                     px_.x,
                     px_.y,
                     idepth_,
                     info_);
}

/// ==========================================================================
void Patch::Extract(const cv::Mat &mat,
                    const Point2dArray &pxs) noexcept
{
    for (int k = 0; k < kSize; k++)
    {
        const auto &px = pxs[k];
        vals[k] = ValAtD<uchar>(mat, px);
        grads[k] = GradAtD<uchar>(mat, px);
    }
}

void Patch::ExtractFast(const cv::Mat &mat,
                        const Point2dArray &pxs) noexcept
{
    for (int k = 0; k < kSize; k++)
    {
        const auto xyv = GradValAtD<uchar>(mat, pxs[k]);
        grads[k].x = xyv[0];
        grads[k].y = xyv[1];
        vals[k] = xyv[2];
    }
}

void Patch::ExtractAround(const cv::Mat &mat,
                          const cv::Point2d &px) noexcept
{
    for (int k = 0; k < kSize; k++)
    {
        const auto &pxk = px + kOffsets[k];
        vals[k] = ValAtD<uchar>(mat, pxk);
        grads[k] = GradAtD<uchar>(mat, pxk);
    }
}

void Patch::ExtractAround2(const cv::Mat &mat,
                          const cv::Point2d &px) noexcept
{
    for (int k = 0; k < kSize; k++)
    {
        const auto &pxk = px + kOffsets[k];
        vals[k] = ValAtD<uchar>(mat, pxk);
        grads[k] = GradAtD2<uchar>(mat, pxk);
    }
}

void Patch::ExtractAround3(const cv::Mat &mat,
                          const cv::Point2d &px) noexcept
{
    for (int k = 0; k < kSize; k++)
    {
        const auto &pxk = px + kOffsets[k];
        vals[k] = ValAtD<uchar>(mat, pxk);
        grads[k] = ScharrAtI<uchar>(mat, pxk);
    }
}

void Patch::ExtractIntensity(const cv::Mat &mat,
                             const cv::Point2d &px) noexcept
{
    for (int k = 0; k < kSize; k++)
    {
        vals[k] = ValAtD<uchar>(mat, pxk);
    }
}

bool Patch::IsAnyOut(const cv::Mat &mat, 
                     const Point2dArray &pxs) noexcept
{
    return std::any_of(std::cbegin(pxs), std::cend(pxs), [&](const auto& px) {
        return IsPixOut(mat, px, border);
    });
}

} // end of namespace adso