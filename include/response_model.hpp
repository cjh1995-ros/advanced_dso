#pragma once

#include <gtest/gtest.h>
#include <absl/types/span.h>
#include <array>

namespace adso
{

enum ResponseModelMode
{
    Linear,
    GrossBergmann
};


class ResponseModel
{
public:
    ResponseModel() = default;
    explicit ResponseModel(ResponseModelMode mode): mode_(mode) 
    {
        for (int i = 0; i < 256; ++i)
        {
            inverse_response_table_[i] = i;
        }
        if (mode_ == ResponseModelMode::Linear)
        {
            grossberg_params_ = {0.0, 0.0, 0.0, 0.0};
        }
        else if (mode_ == ResponseModelMode::GrossBergmann)
        {
            grossberg_params_ = {6.1, 0.0, 0.0, 0.0};
        }
    }

    double RemoveResponse(int o) const noexcept
    {
        EXPECT_GE(o, 0);
        EXPECT_LE(o, 256);
        return inverse_response_table_[o];
    }

    void SetGrossbergParams(absl::Span<const double> params) noexcept
    {
        EXPECT_EQ(params.size(), 4);
        grossberg_params_ = {params[0], params[1], params[2], params[3]};
    }

    void SetInverseResponseContainer(absl::Span<const double> new_inverse) noexcept
    {
        EXPECT_EQ(new_inverse.size(), 256);
        for (int i=0; i<256; ++i)
        {
            inverse_response_table_[i] = 255.0 * new_inverse[i];
        }
    }

    ResponseModelMode GetMode() const noexcept { return mode_; }
    std::array<double, 256> GetInverseResponseTable() const noexcept { return inverse_response_table_; }
    std::array<double, 4> GetGrossbergParams() const noexcept { return grossberg_params_; }

private:
    ResponseModelMode mode_;

    std::array<double, 256> inverse_response_table_;
    std::array<double, 4> grossberg_params_;
};

}