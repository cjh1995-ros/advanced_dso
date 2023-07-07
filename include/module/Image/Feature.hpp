#pragma once

#include "CommonInclude.hpp"

namespace adso
{

class Feature
{
    float x_, y_;
    float radiance_;
    float gradient_;

    std::shared_ptr<Feature> next_feature_; // next feature in the next frame
    std::shared_ptr<Feature> prev_feature_; // previous feature in the previous frame
};

} // namespace adso