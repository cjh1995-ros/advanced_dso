// This is the header file for the KeyFrame class.
#pragma once

#include <absl/types/span.h>
#include <sophus/se3.hpp>

#include <Eigen/Dense>

#include "image.hpp"
#include "util/dim.hpp"
#include "point.hpp"
#include "camera.hpp"

namespace adso
{


struct AffineModel
{
    AffineModel(double a_in = 0, double b_in = 0): ab{a_in, b_in} {}
    Eigen::Vector2d ab{Eigen::Vector2d::Zero()};
    double a() const { return ab[0]; }
    double b() const { return ab[1]; }
};

struct ErrorState
{
    using Vector10d = Eigen::Matrix<double, Dim::kFrame, 1>;
    using Vector10dCRef = Eigen::Ref<const Vector10d>;

    /// @brief  Error state vector
    Vector10d x_{Vector10d::Zero()};

    ErrorState() = default;
    explicit ErrorState(const Vector10dCRef& delta): x_{delta} {}
    const Vector10d& vec() const noexcept { return x_; } // 0~9
    Eigen::Vector2d ab_l() const noexcept { return x_.segment<2>(Dim::kPose); } // 6, 7
    Eigen::Vector2d ab_r() const noexcept { return x_.segment<2>(Dim::kMono); } // 8, 9
    Sophus::SE3d dT() const noexcept
    {
        return {Sophus::SO3d::exp(x_.head<3>()), x_.segment<3>(3)};
    }

    ErrorState& operator+=(const Vector10d& dx) noexcept
    {
        x_ += dx;
        return *this;
    }
};

struct FrameState
{
    FrameState() = default;
    explicit FrameState(Sophus::SE3d tf_w_cl = {},
                        AffineModel affine_l = {},
                        AffineModel affine_r = {}):
        T_w_cl{tf_w_cl}, affine_l{affine_l}, affine_r{affine_r} { }

    /// @brief Main information of frame.
    Sophus::SE3d T_w_cl; // pose of left camera
    AffineModel affine_l; // affine model of left camera
    AffineModel affine_r; // affine model of right camera

    /// @brief Represent frame state as string
    // std::string Repr() const;
    // friend std::ostream& operator<<(std::ostream& os, const FrameState& fs)
    // {
    //     return os << fs.Repr();
    // }

    /// @brief  Update frame state with error state
    FrameState& operator+=(const ErrorState& dx)
    {
        T_w_cl *= dx.dT();
        affine_l.ab += dx.ab_l();
        affine_r.ab += dx.ab_r();
        return *this;
    }
    FrameState& operator-=(const ErrorState& dx)
    {
        T_w_cl *= dx.dT().inverse();
        affine_l.ab -= dx.ab_l();
        affine_r.ab -= dx.ab_r();
        return *this;
    }
    friend FrameState operator+(FrameState st, const ErrorState& dx) noexcept
    {
        return st += dx;
    }
    friend FrameState operator-(FrameState st, const ErrorState& dx) noexcept
    {
        return st -= dx;
    }
};


/// @brief simple frame
struct Frame
{
    using Vector10d = ErrorState::Vector10d;
    using Vector10dCRef = ErrorState::Vector10dCRef;

    // images
    ImagePyramid grays_l_;
    ImagePyramid grays_r_;
    FrameState state_;

    // constructors and deconstructor
    virtual ~Frame() noexcept = default;

    explicit Frame(const ImagePyramid& gray_l,
                   const ImagePyramid& gray_r,
                   const Sophus::SE3d& tf_w_cl,
                   const AffineModel& affine_l = {},
                   const AffineModel& affine_r = {});


    /// @brief Represent frame as string
    // virtual std::string Repr() const;
    // friend std::ostream& operator<<(std::ostream& os, const Frame& frame)
    // {
    //     return os << frame.Repr();
    // }

    /// @brief Information of frame
    int levels() const noexcept { return static_cast<int>(grays_l_.size()); }
    bool empty() const noexcept { return grays_l_.empty(); }
    bool is_stereo() const noexcept { return !grays_r_.empty(); }
    cv::Size image_size() const noexcept 
    {
        if (grays_l_.empty()) return {};
        const auto& mat = grays_l_.front();
        return {mat.cols, mat.rows};
    }

    /// @brief Accessors
    const ImagePyramid& grays_l() const noexcept { return grays_l_; }
    const ImagePyramid& grays_r() const noexcept { return grays_r_; }
    const cv::Mat& gray_l() const noexcept { return grays_l_.front(); }

    FrameState& state() noexcept { return state_; }
    const FrameState& state() const noexcept { return state_; }
    const Sophus::SE3d& Twc() const noexcept { return state_.T_w_cl; }

    /// @brief Modifiers
    void SetGrays(const ImagePyramid& grays_l, const ImagePyramid& grays_r)
    {
        // TODO : check if it is image pyramid and stereo pair
        grays_l_ = grays_l;
        grays_r_ = grays_r;
    };
    void SetTwc(const Sophus::SE3d& tf_w_cl) noexcept { state_.T_w_cl = tf_w_cl; }
    void SetState(const FrameState& state) noexcept { state_ = state; }
    virtual void UpdateState(const Vector10dCRef& dx) noexcept { state_ += ErrorState{dx}; }
};


struct KeyframeStatus
{
    // frame
    int pixels{};
    int depths{};
    int patches{};

    // point0 info
    int info_bad{};
    int info_uncert{};
    int info_ok{};
    int info_max{};

    std::string FrameStatus() const;
    std::string PointStatus() const;
    // std::string TrackStatus() const;

    // std::string Repr() const;
    // friend std::ostream& operator<<(std::ostream& os, const KeyframeStatus& st)
    // {
    //     return os << st.Repr();
    // }

    void UpdateInfo(const FramePointGrid& points0);
};

/// @brief a keyframe is a frame with depth at features
struct Keyframe final : public Frame 
{
    KeyframeStatus status_{};
    FramePointGrid points_{};
    std::vector<PatchGrid> patches_{};  // precomputed patches
    /// @brief whether first estimate is fixed -> marginalization 우선순위를 말함.
    bool fixed_{false};                 
    // error state x in dso paper
    ErrorState x_{};       

    /// @brief Fix first estimate
    bool is_fixed() const noexcept { return fixed_; }
    void SetFixed() noexcept { fixed_ = true; }
    /// @brief This is zeta0 in dso paper
    FrameState GetFirstEstimate() const noexcept;

    /// @brief Update state during optimization, need to call
    /// UpdateLinearizationPoint() to finalize the change
    void UpdateState(const Vector10dCRef& dx) noexcept override;
    void UpdatePoints(const VectorXdCRef& xm, double scale, int gsize = 0);
    void UpdateStatusInfo() noexcept { status_.UpdateInfo(points_); }

    FramePointGrid& points() noexcept { return points_; }
    const KeyframeStatus& status() const noexcept { return status_; }
    const FramePointGrid& points() const noexcept { return points_; }
    const std::vector<PatchGrid>& patches() const noexcept { return patches_; }

    Keyframe() = default;
    // std::string Repr() const override;

    /// @brief Initialize from frame
    void SetFrame(const Frame& frame) noexcept;

    /// @brief Allocate storage for points and patches, not for images
    /// @return number of bytes
    size_t Allocate(int num_levels, const cv::Size& grid_size);
    /// @brief Initialize points (pixels only)
    int InitPoints(const PixelGrid& pixels, const Camera& camera);

    /// @group Initialize point depth from various sources
    int InitFromConst(double depth, double info = SettingPoint::kOkInfo);
    /// @brief Initialize point depth from depths (from RGBD or ground truth)
    // int InitFromDepth(const cv::Mat& depth, double info = SettingPoint::kOkInfo);
    /// @brief Initialize point depth from disparities (from StereoMatcher)
    int InitFromDisp(const cv::Mat& disp,
                    const Camera& camera,
                    double info = SettingPoint::kOkInfo);
    /// @brief Initialize point depth from inverse depths (from FrameAligner)
    int InitFromAlign(const cv::Mat& idepth, double info); // <- 현재는 이것만 쓴다는 가정!

    /// @brief Initialize patches
    /// @return number of patches from all levels
    int InitPatches(int gsize = 0);
    /// @brief Initialize patches at level
    /// @return number of precomputed patches within this level
    int InitPatchesLevel(int level, int gsize = 0);

    /// @brief Reset this keyframe
    void Reset() noexcept;
    bool Ok() const noexcept { return status_.pixels > 0; }
};

using KeyframePtrSpan = absl::Span<Keyframe*>;
using KeyframePtrConstSpan = absl::Span<Keyframe const* const>;

/// @brief Get a reference to the k-th keyframe, with not-null and ok checks
Keyframe& GetKfAt(KeyframePtrSpan keyframes, int k);
const Keyframe& GetKfAt(KeyframePtrConstSpan keyframes, int k);

/// @brief Get the smallest bounding box that covers all points with
/// info >= min_info
cv::Rect2d GetMinBboxInfoGe(const FramePointGrid& points, double min_info);

} // namespace adso