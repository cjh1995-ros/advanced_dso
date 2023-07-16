#pragma once

#include "CommonInclude.hpp"

// #include "Image/ExposureTimeEstimator.hpp"
// #include "Image/ResponseModel.hpp"
// #include "Image/VignetteModel.hpp"
// #include "Image/Feature.hpp"


namespace adso
{

class Frame
{
private:
    cv::Mat original_;
    std::vector<cv::Mat> img_pyramids_;

    std::vector<cv::Size> sizes_;
    std::vector<cv::Mat> gradients_;

    double exposure_time_;

    int idx_; // index of the frame
    int plvl_; // pyramid level

public:
    Frame(cv::Mat orig, int idx, int plvl = 4): 
        original_(orig), plvl_(plvl), idx_(idx) 
        {
            genPyramid();
            genGradientImage();
        };
    
    Frame(cv::Mat orig, int idx, double exp_time, int plvl = 4):
        original_(orig), plvl_(plvl), idx_(idx), exposure_time_(exp_time) {};
    
    ~Frame(){};

    inline void setExposureTime(double exp_time) {exposure_time_ = exp_time;};
    inline void genPyramid() {cv::buildPyramid(original_, img_pyramids_, plvl_);};
    void genGradientImage();
    cv::Size getSize(int lvl) const {return sizes_[lvl];};
};

class StereoFrame
{
private:
    std::unique_ptr<Frame> left_;
    std::unique_ptr<Frame> right_;

    std::shared_ptr<StereoFrame> p_ref_;

    // make pixel selector for left frame

public:
    StereoFrame(cv::Mat left, cv::Mat right, int idx, int plvl = 4):
        left_(std::make_unique<Frame>(left, idx, plvl)),
        right_(std::make_unique<Frame>(right, idx, plvl)) {};
    
    StereoFrame(std::vector<cv::Mat> images, int idx, int plvl = 4):
        left_(std::make_unique<Frame>(images[0], idx, plvl)),
        right_(std::make_unique<Frame>(images[1], idx, plvl)) {};
    
    bool isSameSize(int lvl) const {return left_->getSize(lvl) == right_->getSize(lvl);};
};

} // namespace adso