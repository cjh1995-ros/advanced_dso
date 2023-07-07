#pragma once

#include "CommonInclude.hpp"

#include "Image/ExposureTimeEstimator.hpp"
#include "Image/ResponseModel.hpp"
#include "Image/VignetteModel.hpp"
#include "Image/Feature.hpp"


namespace adso
{

class Frame
{
private:
    cv::Mat original_;
    cv::Mat radiance_;
    std::vector<std::unique_ptr<Eigen::Vec3f> > vp_gradient_;

    // features in the frame. policy is from DSO
    std::vector<std::unique_ptr<Feature> > vp_features_;
    
    // vignette and response model. usually set before estimate exposure time.
    std::unique_ptr<VigentteModel> p_vignette_;
    std::unique_ptr<ResponseModel> p_response_;
    double exposure_time_;

    int idx_; // index of the frame
    int plvl_; // pyramid level

public:
    Frame(cv::Mat orig, int idx, int plvl = 4): 
        original_(orig), plvl_(plvl), idx_(idx) {};
    
    Frame(cv::Mat orig, ind idx, double exp_time, int plvl = 4):
        original_(orig), plvl_(plvl), idx_(idx), exposure_time_(exp_time) {};
    
    ~Frame();

    inline void setExposureTime(double exp_time) { exposure_time_ = exp_time; };
    inline void setVignette(std::unique_ptr<VignetteModel> vignette) 
        { p_vignette_ = std::move(vignette); };
    inline void setResponse(std::unique_ptr<ResponseModel> response) 
        { p_response_ = std::move(response); };

    inline void genRadianceImage(){};



};


} // namespace adso