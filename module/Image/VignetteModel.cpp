#include "ImageIntensity/VignetteModel.hpp"

namespace adso
{

template <typename T>
T VigentteModel<T>::getVignetteFactor(const cv::Point2f pixel)
{
    T radius = getNormalizedRadius(pixel);
    return getVignetteFactor(radius);
}

template <typename T>
T VigentteModel<T>::getVignetteFactor(const T norm_radius)
{
    T r2 = norm_radius * norm_radius;
    T r4 = r2 * r2;
    T r6 = r4 * r2;
    T factor = 1.0 + v1 * r2 + v2 * r4 + v3 * r6;
    return factor;
}

} // namespace adso