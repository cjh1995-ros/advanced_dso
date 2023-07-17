#pragma once

#include "Image/Grid.hpp"
#include "Eigen.hpp"
#include "Dimension.hpp"


namespace adso
{


struct DepthPoint
{
    static constexpr double kBadInfo = -1.0;
    static constexpr double kMinInfo = 0.0; // infinite meter
    static constexpr double kOkInfo = 5.0;
    static constexpr double kMaxInfo = 10.0; // 0.1 meter
    static constexpr double kBadIdepth = -1.0;
    static constexpr double kNanD = std::numeric_limits<double>::quiet_NaN();
    inline static const cv::Point2d kBadPixD = {kNanD, kNanD};

    /// @brief Main data in this structure. But info_ is mutable.
    cv::Point2d px_{kBadPixD};
    double idepth_{kBadIdepth};
    mutable double info_{kBadInfo};
 
    Eigen::Vector2d uv() const noexcept {return {px_.x, px_.y};}
    const cv::Point2d& px() const noexcept {return px_;}
    double idepth() const noexcept {return idepth_;}
    double info() const noexcept {return info_;}

    /// @brief Point is not selected by PixelSelector 
    bool PixelBad() const {return std::isnan(px_.x) || std::isnan(px_.y);}
    bool PixelOk() const {return !PixelBad();}

    /// @brief Point has no valid depth
    bool DepthBad() const {return idepth_ < 0.0;}
    bool DepthOk() const {return idepth_ >= 0.0;}

    bool InfoBad() const {return info_ < kMinInfo;}
    bool InfoOk() const {return info_ >= kMinInfo;}
    bool InfoMax() const {return info_ >= kMaxInfo;}

    /// @brief Point is skipped for initialization of depth. 
    ///         If it has good depth or bad pixel value, it is skipped.
    bool SkipInit() const noexcept {return DepthOk() || PixelBad();}

    /// @brief Point is skipped for alignment. Allow points with good info 
    bool SkipAlign() const noexcept {return !InfoOk() || DepthBad() || PixelBad();}

    /// @brief Modifiers
    void SetPix(const cv::Point2d &px) noexcept {px_ = px;}
    void SetIdepthInfo(double idepth, double info) noexcept;
    void UpdateIdepth(double d_idepth) noexcept {idepth_ = std::max(0.0, idepth_ + d_idepth);}
    void UpdateInfo(double d_info) noexcept {info_ = std::max(kMinInfo, info_ + d_info);}


    std::string Repr() const;
    friend std::ostream& operator<<(std::ostream &os, const DepthPoint &p) {return os << p.Repr();}
};

/// @brief Idepth point at a keyframe, stores jacobian
struct FramePoint final: public DepthPoint
{
    using Matrix23d = Eigen::Matrix<double, 2, 3>;
    using Matrix26d = Eigen::Matrix<double, 2, 6>;
    static constexpr int kBadHid = -1;

    /// @brief Main point information. It has id and pixel position in normalized coordinate
    mutable int hid_{kBadHid}; // hessian id used in PBA, also used in align
    Eigen::Vector2d nc{0, 0};  // normalized coordinate

    /// @brief get 3d point, assumes idepth is not 0, otherwise UB 
    Eigen::Vector3d pt() const noexcept {return nh() / idepth_;}
    /// @brief Get homogeneous normalized image coordinate
    Eigen::Vector3d nh() const noexcept {return {nc.x(), nc.y(), 1.0};}

    void SetNc(const Eigen::Vector3d& nh) noexcept{ nc = nh.head<2>(); }
    bool HidBad() const noexcept {return hid < 0;}
};


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

    // Pixel offsets of each pixel in patch (first is center, no offset)
    //         [0, -1]
    // [-1, 0] [0,  0] [1, 0]
    //         [0,  1]
    inline static const Point2dArray kOffsetPx = {
        cv::Point2d{0, 0}, {0, -1}, {-1, 0}, {1, 0}, {0, 1}};

    ArrayKd vals{};        // raw image intensity values
    Point2dArray grads{};  // raw image gradients


    /// @brief Whether this patch is valid
    bool Bad() const noexcept {return vals[kCenter] < 0.0;}
    bool SetBad() noexcept {vals[kCenter] = -1.0;}
    bool Ok() const noexcept {return !Bad();}

    static auto offsets() noexcept {return Eigen::Map<const Matrix2Kd> (&kOffsetPx[0].x);}

    auto gxys() const noexcept { return Eigen::Map<const Matrix2Kd> (&grads[0].x);}

    ArrayKd GradSqNorm() const noexcept {return gxys().colwise().squaredNorm().transpose();}

    /// @brief Extract intensity and gradient from gray image at patch pxs
    void Extract(const cv::Mat &img, const Point2dArray &pxs) noexcept;
    void ExtractFast(const cv::Mat& image, const Point2dArray& pxs) noexcept;
    void ExtractAround(const cv::Mat& image, const cv::Point2d& px) noexcept;
    void ExtractAround2(const cv::Mat& image, const cv::Point2d& px) noexcept;
    void ExtractAround3(const cv::Mat& image, const cv::Point2d& px) noexcept;
    void ExtractIntensity(const cv::Mat& image, const Point2dArray& pxs) noexcept;

    /// @brief If any px is OOB
    static bool IsAnyOut(const cv::Mat &img,
                         const Point2dArray &pxs,
                         double border) noexcept;
};

using PatchGrid = Grid2d<Patch>;
using PixelGrid = Grid2d<cv::Point2i>;
using DepthPointGrid = Grid2d<DepthPoint>;
using FramePointGrid = Grid2d<FramePoint>;


} // namespace adso