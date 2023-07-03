#pragma once

#include "CommonInclude.hpp"

namespace adso
{


class ImageAndIrradiance
{
private:
    cv::Mat input_image; // raw color image
    cv::Mat output_irradiance; // raw irradiance image
    

public:
    ImageAndIrradiance(cv::Mat input_image, cv::Mat output_irradiance);
    ~ImageAndIrradiance();

    inline void readPhotometricCalibFiles(
        const std::string& response_file,
        const std::string& vignetting_file
    )

    cv::Mat getInputImage();
    cv::Mat getOutputIrradiance();

};


} // namespace adso