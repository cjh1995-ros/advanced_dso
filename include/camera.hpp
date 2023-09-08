#pragma once

#include <opencv2/core/mat.hpp>
#include "util/eigen.hpp"

namespace adso
{

/// @brief Scale fxycxy 
Eigen::Array4d ScaleFxycxy(const Eigen::Array4d& fxycxy, double scale) noexcept;

/// @brief Jacobian of projection wrt point
MatrixMNd<2, 3> DprojDpoint(const Eigen::Vector3d& pt) noexcept;

/// @brief Convert from level to scale
constexpr double PyrLevel2Scale(int level) noexcept
{
    return std::pow(2.0, -level);
}

/// @brief PinholeCamera
struct Camera
{
    // Basic infos
    cv::Size size_{};
    Eigen::Array4d fxycxy_{Eigen::Array4d::Zero()};
    double baseline_{};
    double scale_{1.0};


    Camera() = default;
    Camera(const cv::Size& size,
           const Eigen::Array4d& fxycxy,
           double baseline = 0.0,
           double scale = 1.0);

    /// @brief Create camera from a mat of size 1x5
    static Camera FromMat(const cv::Size& size, const cv::Mat& intrinsic);

    /// @brief Repr / <<
    std::string Repr() const;
    friend std::ostream& operator<<(std::ostream& os, const Camera& cam)
    {
        return os << cam.Repr();
    }

    /// @brief Returns a scaled camera
    Camera Scaled(double scale) const noexcept;
    /// @brief Returns a camera at pyramid level
    Camera AtLevel(int level) const noexcept
    {
        return Scaled(PyrLevel2Scale(level));
    }

    /// @brief Accessors
    double fx() const noexcept { return fxycxy_[0]; }
    double fy() const noexcept { return fxycxy_[1]; }
    double cx() const noexcept { return fxycxy_[2]; }
    double cy() const noexcept { return fxycxy_[3]; }
    Eigen::Array2d fxy() const noexcept { return fxycxy_.head<2>(); }
    Eigen::Array2d cxy() const noexcept { return fxycxy_.tail<2>(); }
    const Eigen::Array4d& fxycxy() const noexcept { return fxycxy_; }

    const cv::Size& cvsize() const noexcept { return size_; }
    int height() const noexcept { return size_.height; }
    int width() const noexcept { return size_.width; }

    double scale() const noexcept { return scale_; }
    double baseline() const noexcept { return baseline_; }
    bool Ok() const noexcept { return size_.area() > 0; }
    bool is_stereo() const noexcept { return baseline_ > 0; }

    /// @brief Project point to image
    template <int N>
    MatrixMNd<2, N> Forward(const MatrixMNd<3, N>& pts) const noexcept
    {
        auto uv = ((pts.template topRows<2>()).array().rowwise() / 
                    pts.template bottomRows<1>().array())
                        .eval(); 
        uv.colwise() *= fxycxy_.head<2>();
        uv.colwise() += fxycxy_.tail<2>();
        return uv;
    }

    /// @brief Backproject image to point
    template <int N>
    MatrixMNd<3, N> Backward(const MatrixMNd<2, N>& uv) const noexcept
    {
        auto nc = (uv.array().colwise() - fxycxy_.tail<2>()).eval();
        nc.colwise() /= fxycxy_.head<2>();
        MatrixMNd<3, N> nh;
        nh.template topRows<2>() = nc;
        nh.template bottomRows<1>().setOnes();
        return nh;
    }

    /// @brief Convert inverse depth to disparity
    template <int M, int N>
    ArrayMNd<M, N> Idepth2Disp(const ArrayMNd<M, N>& idepth) const noexcept
    {
        return fx() * baseline() * idepth;
    }

    /// @brief Convert disparity to inverse depth
    double Depth2Disp(double depth) const noexcept
    {
        return fx() * baseline() / depth;
    }

    /// @brief Convert disparity to inverse depth
    double Disp2Idepth(double disp) const noexcept
    {
        return disp / (fx() * baseline());
    }
};


/// @brief Brown-Conrady model
struct BrownConrady : public Camera
{
    // Eigen::Array4d k_{Eigen::Array4d::Zero()}; // k1, k2, p1, p2

    // BrownConrady() = default;
    // BrownConrady(const cv::Size& size,
    //                 const Eigen::Array4d& fxycxy,
    //                 const Eigen::Array4d& k,
    //                 double baseline = 0.0,
    //                 double scale = 1.0);

    // /// @brief Create camera from a mat of size 1x9
    // static BrownConrady FromMat(const cv::Size& size, const cv::Mat& intrinsic);

    // /// @brief Repr / <<
    // std::string Repr() const override;
    // friend std::ostream& operator<<(std::ostream& os, const BrownConrady& cam)
    // {
    //     return os << cam.Repr();
    // }

    // /// @brief Accessors
    // const Eigen::Array4d& k() const noexcept { return k_; }

    // /// @brief Project point to image
    // template <int N>
    // MatrixMNd<2, N> Forward(const MatrixMNd<3, N>& pts) const noexcept override
    // {
    //     auto uv = ((pts.template topRows<2>()).array().rowwise() / 
    //                 pts.template bottomRows<1>().array())
    //                     .eval(); 
    //     uv.colwise() *= fxycxy_.head<2>();
    //     uv.colwise() += fxycxy_.tail<2>();
    //     return Undistort(uv);
    // }

    // /// @brief Backproject image to point
    // template <int N>
    // MatrixMNd<3, N> Backward(const MatrixMNd<2, N>& uv) const noexcept override
    // {
    //     auto nc = Distort(uv);
    //     nc.colwise() -= fxycxy_.tail<2>();
    //     nc.colwise() /= fxycxy_.head<2>();
    //     MatrixMNd<3, N> nh;
    //     nh.template topRows<2>() = nc;
    //     nh.template bottomRows<1>().setOnes();
    //     return nh;
    // }

    // /// @brief Convert inverse depth to disparity
    // template <int M, int N>
    // ArrayMNd<M, N> Idepth2Disp(const ArrayMNd<M, N>& idepth) const noexcept override
    // {
    //     return fx() * baseline() * idepth;
    // }
};


/// @brief Kannala-Brandt model
struct KannalaBrandt : public Camera
{

};


/// @brief Double sphere model
struct DoubleSphere : public Camera
{

};




/// @brief Project 3d -> 2d (single/batch)
template <int N>
MatrixMNd<2, N> Project(const MatrixMNd<3, N>& pt) noexcept
{
    return (pt.template topRows<2>()).array().rowwise() /
            (pt.template bottomRows<1>()).array();
}

/// @brief Homogenize by appending 1
template <int N>
MatrixMNd<3, N> Homogenize(const MatrixMNd<2, N>& v) noexcept 
{
    MatrixMNd<3, N> h;
    h.template topRows<2>() = v;
    h.template bottomRows<1>().setOnes();
    return h;
}

/// @brief Convert from pixel to normalized image coordinate (single/batch)
template <int N>
MatrixMNd<2, N> PnormFromPixel(const MatrixMNd<2, N>& uv,
                               const Eigen::Array4d& fc) noexcept 
{
    return (uv.array().colwise() - fc.tail<2>()).colwise() / fc.head<2>();
}

/// @brief Convert from normalized image coordinate to pixel
template <int N>
MatrixMNd<2, N> PixelFromPnorm(const MatrixMNd<2, N>& nc,
                               const Eigen::Array4d& fc) noexcept 
{
    return (nc.array().colwise() * fc.head<2>()).colwise() + fc.tail<2>();
}

/// @brief Scale pixel assuming center of top-left corner is (0, 0)
template <int N>
MatrixMNd<2, N> ScaleUv(const MatrixMNd<2, N>& uv, double scale) noexcept 
{
    return (uv.array() + 0.5) * scale - 0.5;
}
} // namespace adso