#pragma once

#include "CommonInclude.hpp"

namespace adso
{

class Jacobian8
{
private:
    Vector8f data_;
    
    float fx_, fy_, cx_, cy_;
    std::vector<Eigen::Vector3f> host_points_;

public:
    Jacobian8(const float fx, const float fy, const float cx, const float cy)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy)
    {
        data_.setZero();
    };

    inline void getHessian(Matrix88f& matrix88f)
    {
        
    };
};

class DerivationTargetPhoto
{
    
};

class JacobianPhotometric
{
    
};




} // end of adso