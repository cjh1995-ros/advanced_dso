#include "response_model.hpp"
#include <gtest/gtest.h>
#include <algorithm>

namespace adso 
{

TEST(TestResponseModel, TestInitResponseModelGrossBergmann)
{
    ResponseModel rm(ResponseModelMode::GrossBergmann);

    std::array<double, 4> gbp = {6.1, 0.0, 0.0, 0.0};
    std::array<double, 256> irt;
    for (int i=0; i<256; ++i)
    {
        irt[i] = i;
    }

    EXPECT_EQ(rm.GetMode(), ResponseModelMode::GrossBergmann);
    EXPECT_EQ(rm.GetInverseResponseTable(), irt);
    EXPECT_EQ(rm.GetGrossbergParams(), gbp);
}

TEST(TestResponseModel, TestInitResponseModelLinear)
{
    ResponseModel rm(ResponseModelMode::Linear);

    std::array<double, 4> gbp = {6.1, 0.0, 0.0, 0.0};
    std::array<double, 4> gbp_zero = {0.0, 0.0, 0.0, 0.0};
    std::array<double, 256> irt;
    for (int i=0; i<256; ++i)
    {
        irt[i] = i;
    }

    EXPECT_EQ(rm.GetMode(), ResponseModelMode::Linear);
    EXPECT_EQ(rm.GetInverseResponseTable(), irt);
    EXPECT_NE(rm.GetGrossbergParams(), gbp);
    EXPECT_EQ(rm.GetGrossbergParams(), gbp_zero);
}

TEST(TestResponseModel, TestRemoveResponse)
{
    ResponseModel rm(ResponseModelMode::GrossBergmann);

    std::array<double, 4> gbp = {6.1, 0.0, 0.0, 0.0};

    int o = 200;

    EXPECT_DOUBLE_EQ(rm.RemoveResponse(o), 200.0);
}

TEST(TestResponseModel, TestSetInverseResponseContainer)
{
    ResponseModel rm(ResponseModelMode::GrossBergmann);

    std::array<double, 256> new_array;

    for (int i=256; i>=0; --i)
    {
        new_array[i] = static_cast<double>(255.0 - i) / 255.0;
    }

    rm.SetInverseResponseContainer(new_array);

    EXPECT_EQ(rm.RemoveResponse(0), 255.0);
    EXPECT_EQ(rm.RemoveResponse(255), 0.0);
    EXPECT_EQ(rm.RemoveResponse(128), 127.0);
}



} // namespace adso