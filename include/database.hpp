#pragma once
#include <opencv2/core.hpp>
#include "keyframe.hpp"
#include "vignette_model.hpp"
#include "response_model.hpp"

#include <vector>


namespace adso
{

class Database
{
public:
    Database() = default;
    explicit Database(const cv::Size& size);

    //////////////// Handling keyframe ///////////////////////
    /// @brief add keyframe to database
    void AddKeyFrame(const KeyFrame& frame) { frames_.push_back(frame); }

    /// @brief get keyframe from database
    const KeyFrame& GetKeyFrame(int idx) const { return frames_[idx]; }

    /// @brief remove keyframe from database
    void RemoveKeyFrame(int idx) { frames_.erase(frames_.begin() + idx); }

    /// @brief remove last frame from database
    void RemoveOldFrame() { frames_.pop_front(); }

    //////////////// Handling Vignette ///////////////////////
    void SetVignetteModel(const VignetteModel& vig) { vignette_model_ = vig; }
    const VignetteModel& GetVignetteModel() const { return vignette_model_; }

    //////////////// Handling Response ///////////////////////
    void SetResponseModel(const ResponseModel& res) { response_model_ = res; }
    const ResponseModel& GetResponseModel() const { return response_model_; }

private:
    cv::Size size_;
    std::vector<KeyFrame> frames_;
    VignetteModel vignette_model_;
    ResponseModel response_model_;
};




} // namespace adso