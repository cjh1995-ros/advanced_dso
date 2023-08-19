// This is the header file for the KeyFrame class.
#pragma once

#include <opencv2/core.hpp>
#include "image.hpp"

namespace adso
{

class KeyFrame
{

private:
    ImagePyramid colors_l_;
    ImagePyramid colors_r_;
    ImagePyramid grays_l_;
    ImagePyramid grays_r_;

};

} // namespace adso