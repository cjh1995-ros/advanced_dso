#include "point.hpp"

#include "util/logging.hpp"
#include "util/pixel_operate.hpp"



namespace adso
{

void DepthPoint::SetIdepthInfo(double idepth, double info) noexcept
{
    CHECK(PixelOk());
    CHECK_GE(idepth, 0);
    CHECK_LE(info, cfg_.kMaxInfo);
    idepth_ = idepth;
    info_ = info;
}

std::string DepthPoint::Repr() const
{
    return fmt::format("DepthPoint(uv = ({}, {}), idepth={:.04f}, info={})",
        px_.x, px_.y, idepth_, info_);
}


void Patch::Extract(const cv::Mat& mat, const Point2dArray& pxs) noexcept
{
    for (int k=0; k<Patch::kSize; ++k)
    {
        const auto xyv = GradValAtD<uchar>(mat, pxs[k]);
        grads_[k].x = xyv.x;
        grads_[k].y = xyv.y;
        vals_[k] = xyv.z;
    }
}

void Patch::ExtractAround(const cv::Mat& image,
                          const cv::Point2d& px) noexcept 
{
    for (int k = 0; k < kSize; ++k) 
    {
        const auto px_k = px + kOffsetPx[k];
        vals_[k] = ValAtD<uchar>(image, px_k);
        grads_[k] = GradAtD<uchar>(image, px_k);
    }
}


void Patch::ExtractIntensity(const cv::Mat& mat,
                             const Point2dArray& pxs) noexcept
{
    for (int k=0; k<Patch::kSize; ++k)
    {
        vals_[k] = ValAtD<uchar>(mat, pxs[k]);
    }
}

bool Patch::IsAnyOut(const cv::Mat& mat,
                     const Point2dArray& pxs,
                     double border) noexcept
{
    return std::any_of(std::cbegin(pxs), std::cend(pxs),
                       [&](const auto& px) {
                           return IsPixOut(mat, px, border);
                       });
}

} // namespace adso