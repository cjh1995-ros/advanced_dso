#pragma once

#include "CommonInclude.hpp"

namespace adso
{

class ExposureTimeEstimator
{
private:
    /// @brief How many frames do we use for estimating exposure time
    int window_size_;

    /// @brief The database who keeps the frames
    Database* database_;

public:
    /// @brief Construct a new Exposure Estimator object
    /// @param database 
    /// @param window_size 
    ExposureTimeEstimator(Database* database, int window_size): 
        database_(database), window_size_(window_size) {};

    /// @brief estimate the exposure time. It is qucik version with 
    /// restricted window size
    /// @return exposure time
    double estimateExposureTimeQuick();

    /// @brief estimate the exposure time with all the frames in the database
    /// @return exposure time
    double estimateExposureTimeSlow();
};

} // namespace adso