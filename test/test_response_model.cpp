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
    std::array<double, 4> gbp_real = {0.0, 0.0, 0.0, 0.0};
    std::array<double, 256> irt;
    for (int i=0; i<256; ++i)
    {
        irt[i] = i;
    }

    EXPECT_EQ(rm.GetMode(), ResponseModelMode::Linear);
    EXPECT_EQ(rm.GetInverseResponseTable(), irt);
    EXPECT_NE(rm.GetGrossbergParams(), gbp);
    EXPECT_EQ(rm.GetGrossbergParams(), gbp_real);
}

} // namespace adso

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); // Google Test를 실행하고 결과를 저장합니다.
}
