#pragma once

#include "CommonInclude.hpp"

namespace adso
{

template <typename T>
class VigentteModel
{
private:
    int image_width;
    int image_height;
    
    T v1;
    T v2;
    T v3;
    T max_radius;
public:
    VigentteModel(T v1_, T v2_, T v3_, int w_, int h_):
        v1(v1_), v2(v2_), v3(v3_), image_width(w_), image_height(h_)
    {
        T w2 = (T)image_width / 2.0;
        T h2 = (T)image_height / 2.0;
        max_radius = sqrt(w2 * w2 + h2 * h2);
    }

    inline T getNormalizedRadius(const cv::Point2f pixel)
    {
        T x = (T)pixel.x - image_width / 2.0;
        T y = (T)pixel.y - image_height / 2.0;
        T radius = sqrt(x*x + y*y);
        return radius/max_radius;
    }

    inline std::vector<T> getVignetteEsimated()
    {
        std::vector<T> tmp;
        tmp.push_back(v1);
        tmp.push_back(v2);
        tmp.push_back(v3);
        return tmp;
    }

    inline void setVignetteParameters(const std::vector<T> &vignette_params)
    {
        v1 = vignette_params[0];
        v2 = vignette_params[1];
        v3 = vignette_params[2];
    }

    T getVignetteFactor(const cv::Point2f pixel);
    T getVignetteFactor(const T norm_radius)
};



} // namespace adso