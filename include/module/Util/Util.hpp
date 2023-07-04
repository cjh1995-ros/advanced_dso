#include "CommonInclude.hpp"

EIGEN_ALWAYS_INLINE float getInterpolatedElement(
    const float* const mat, 
    const float x,
    const float y,
    const int width)
{
    int ix = (int) x;
    int iy = (int) y;

    float dx = x - ix;
    float dy = y - iy;

    float dxdy = dx * dy;
    const float* bp = mat + ix + iy * width;

    float res = dxdy * bp[1 + width]

}