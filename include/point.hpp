#pragma once

#include "util/dim.hpp"
#include "util/grid.hpp"
#include "util/pixel_operate.hpp"


namespace adso
{

struct SettingPoint
{
    static constexpr double kBadInfo = -1.0;
    static constexpr double kMinInfo = 0.0;
    static constexpr double kOkInfo = 5.0;
    static constexpr double kMaxInfo = 10.0;
    static constexpr double kBadIdepth = -1.0;
    static constexpr double kNand = std::numeric_limits<double>::quiet_NaN();
    inline static const cv::Point2d kBadPixD = {kNand, kNand};
    static constexpr int kBadHid = -1;
};


/// @brief Pixel and Idepth value
struct DepthPoint
{
private:
    SettingPoint cfg_; // setting for point
    cv::Point2d px_; // pixel coordinate
    double idepth_; // inverse depth
    mutable double info_; // jacobian value

public:
    DepthPoint(SettingPoint cfg = SettingPoint()): cfg_(cfg), px_(cfg_.kBadPixD), idepth_(cfg_.kBadIdepth), info_(cfg_.kBadInfo) {}

    /// @brief return basic data of point
    Eigen::Vector2d uv() const noexcept { return Eigen::Vector2d(px_.x, px_.y); }
    const cv::Point2d& px() const noexcept { return px_; }
    double idepth() const noexcept { return idepth_; }
    double info() const noexcept { return info_; }

    /// @brief check good or bad 
    bool PixelBad() const {return std::isnan(px_.x) || std::isnan(px_.y);}
    bool PixelOk() const {return !PixelBad();}

    /// @brief check point has no valid depth
    bool DepthBad() const noexcept {return idepth_ < 0;}
    bool DepthOk() const noexcept {return idepth_ >= 0;}

    /// @brief check point has good or bad info
    bool InfoBad() const noexcept {return info_ < cfg_.kMinInfo;}
    bool InfoOk() const noexcept {return info_ >= cfg_.kOkInfo;}
    bool InfoMax() const noexcept {return info_ == cfg_.kMaxInfo;}

    /// @brief skip initialization if point good depth or pixel is bad
    bool SkipInit() const noexcept {return DepthOk() || PixelBad();}

    /// @brief point is skipped for alignment. allow only good info
    bool SkipAlign() const noexcept {return !InfoOk() || PixelBad() || DepthBad();}

    /// @brief Modifiers
    void SetPix(const cv::Point2d& px) noexcept {px_ = px;}
    void UpdateIdepth(double d_idepth) noexcept
    {
        idepth_ = std::max(0.0, idepth_ + d_idepth);
    }
    void UpdateInfo(double d_info) const noexcept
    {
        info_ = std::min(cfg_.kMaxInfo, info_ + d_info);
    }
    /// @brief 
    void SetIdepthInfo(double idepth, double info) noexcept;

    std::string Repr() const;
    friend std::ostream& operator<<(std::ostream& os, const DepthPoint& p)
    {
        return os << p.Repr();
    }
};


/// @brief Idepth point at a keyframe, stores jacobian.
struct FramePoint final : public DepthPoint
{

    using Matrix23d = Eigen::Matrix<double, 2, 3>;
    using Matrix26d = Eigen::Matrix<double, 2, 6>;

    mutable int hid_ = -1; // hessian id for photometric bundle adjustment
    Eigen::Vector2d nc{0, 0}; // normalized coordinate

    
    Eigen::Vector3d pt() const noexcept {return nh() / idepth();}
    Eigen::Vector3d nh() const noexcept {return {nc.x(), nc.y(), 1.0};}

    bool HidBad() const noexcept {return hid_ < 0;}

    void SetNc(const Eigen::Vector3d& nh) noexcept {nc = nh.head<2>();}
    int hid() const noexcept {return hid_;}
    void SetHid(int hid) noexcept {hid_ = hid;}
};


struct SettingPatch
{
    static constexpr int kSize = Dim::kPatch;
    static constexpr int kCenter = 0;
    static constexpr int kBorder = 2;
};


/// @brief Patch with 4 extra pixels around center
struct Patch
{
    static constexpr int kSize = Dim::kPatch;
    static constexpr int kCenter = 0;
    static constexpr int kBorder = 2;
    
    // Types
    using Point2dArray = std::array<cv::Point2d, kSize>;
    using Matrix2Kd = Eigen::Matrix<double, 2, kSize>;
    using MatrixK2d = Eigen::Matrix<double, kSize, 2>;
    using ArrayKd = Eigen::Array<double, kSize, 1>;

    inline static const Point2dArray kOffsetPx = {
        cv::Point{0, 0}, {0, -1}, {-1, 0}, {1, 0}, {0, 1}
    };

    // Data
    ArrayKd vals_{};         // raw image intensity values
    Point2dArray grads_{};   // raw image gradients


    /// @brief Whether this patch is good 
    bool Bad() const noexcept {return vals_[kCenter] < 0;}
    void SetBad() noexcept {vals_[kCenter] = -1;}
    bool Ok() const noexcept {return !Bad();}

    static auto offsets() noexcept
    {
        return Eigen::Map<const Matrix2Kd>(&kOffsetPx[0].x);
    }

    /// @brief Access to  
    auto gxys() const noexcept
    {
        return Eigen::Map<const Matrix2Kd>(&grads_[0].x);
    }

    ArrayKd GradSqNorm() const noexcept
    {
        return gxys().colwise().squaredNorm().transpose();
    }

    /// @brief Extract intensity and gradient from gray image at patch pxs
    void Extract(const cv::Mat& image, const Point2dArray& pxs) noexcept;
    void ExtractAround(const cv::Mat& image, const cv::Point2d& px) noexcept;
    // void ExtractFast(const cv::Mat& image, const Point2dArray& pxs) noexcept;
    void ExtractIntensity(const cv::Mat& image, const Point2dArray& pxs) noexcept;

    static bool IsAnyOut(const cv::Mat& mat, 
                         const Point2dArray& pxs, 
                         double border) noexcept;
};

/// @brief Diverse types in grids
using PatchGrid = Grid2d<Patch>;
using PixelGrid = Grid2d<cv::Point2i>;
using DepthPointGrid = Grid2d<DepthPoint>;
using FramePointGrid = Grid2d<FramePoint>;

} // namespace adso