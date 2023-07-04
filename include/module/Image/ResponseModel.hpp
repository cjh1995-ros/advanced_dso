#pragma once

#include "CommonInclude.hpp"

namespace adso
{

template <typename T>
class ResponseModel
{
private:
    std::vector<T> grossberg_parameters;
    std::vector<T> inv_responsevec;

public:
    ResponseModel()
    {
        this->inv_responsevec.clear();
        for(int i = 0;i < 256;i++)
        {
            this->inv_responsevec.push_back(i);
        }

        this->grossberg_parameters.clear();
        this->grossberg_parameters.push_back(6.1);
        this->grossberg_parameters.push_back(0.0);
        this->grossberg_parameters.push_back(0.0);
        this->grossberg_parameters.push_back(0.0);
    }

    inline void setInverseResponseVector(std::vector<T> *new_inverse)
    {
        this->inv_responsevec = new_inverse;
    }

    inline void setInverseResponseVector(T* new_inverse)
    {
        for(int i = 0;i < 256;i++)
        {
            this->inv_responsevec.at(i) = 255.0 * new_inverse[i];
        }
    }

    inline std::vector<T> getResponseEstimate()
    {
        return this->grossberg_parameters;
    }

    inline void setGrossbergParameterVector(std::vector<T> params)
    {
        this->grossberg_parameters = params;
    }

    inline T removeResponse(int o)
    {
        return m_inverse_response_vector.at(o);
    }

};


} // namespace adso
