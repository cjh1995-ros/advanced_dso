#pragma once

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

// #include "frame.hpp"
#include "vignette_model.hpp"
#include "response_model.hpp"

#include <memory>
#include <vector>
#include <array>



namespace adso
{

struct DatabaseCfg
{

};


class Database
{
    using ResponseModelParams = std::array<double, 4>;
    using ResponseModelInvParams = std::array<double, 256>;

public:
    Database() = default;
    explicit Database(const DatabaseCfg& cfg){}

    //////////////// Handling frame ///////////////////////
    /// @brief add frame to database
    void AddFrame(const cv::Mat& frame) { frame_history_.push_back(frame); }
    // Sophus::SE3d GetLastLeftPose() const noexcept { return frame_history_.back().T_w_cl; }
    
    /// @brief add if frames is empty
    // Sophus::SE3d GetLastIntervalPose() const noexcept 
    // {
    //     auto& const last_pose = frame_history_.back().T_w_cl;
    //     auto& const last_last_pose = frame_history_.end()[-2].T_w_cl;
    //     return last_pose * last_last_pose.inverse();
    // }

    //////////////// Handling keyframe ///////////////////////
    /// @brief add keyframe to database
    void AddKeyFrame(const cv::Mat& frame) { keyframes_.push_back(frame); }

    /// @brief get keyframe from database
    const cv::Mat& KeyframeAt(int at) const { return keyframes_[at]; }

    /// @brief remove keyframe from database
    void RemoveKeyFrameAt(int at) { keyframes_.erase(keyframes_.begin() + at); }

    /// @brief remove last frame from database
    // void RemoveOldKeyFrame() { keyframes_.pop_front(); }
    /////////////////// End of Handling Datas ///////////////////////

    /////////////////// Getter & Setter ///////////////////////
    cv::Size get_size() const noexcept { return size_; }
    int get_n_frames() const noexcept { return frame_history_.size(); }
    int get_n_keyframes() const noexcept { return keyframes_.size(); }
    std::vector<double> get_vignette_factors() const noexcept { return vignette_model_->GetVigneeteEstimate(); }
    ResponseModelParams get_response_params() const noexcept { return response_model_->GetResponseParams(); }
    ResponseModelInvParams get_response_inv_params() const noexcept { return response_model_->GetInverseResponseTable(); }
    int get_n_optimize_frames() const noexcept { return n_optimize_frames_; }

    void set_size(const cv::Size& size) noexcept { size_ = size; }
    void set_vignette_params(const std::vector<double>& params) noexcept { vignette_model_->SetVignetteParameters(params); }
    void set_n_optimize_frames(int n) noexcept { n_optimize_frames_ = n; }


private:
    cv::Size size_;
    std::vector<cv::Mat> frame_history_;
    std::vector<cv::Mat> keyframes_;
    std::unique_ptr<VignetteModel> vignette_model_;
    std::unique_ptr<ResponseModel> response_model_;

    int n_optimize_frames_ = 50;
    DatabaseCfg cfg_;
};




} // namespace adso