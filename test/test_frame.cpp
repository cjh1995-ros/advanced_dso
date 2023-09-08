#include "frame.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Dense>
#include <sophus/se3.hpp>



namespace adso
{

TEST(TestFrame, TestAffineModel)
{
    AffineModel model;
    EXPECT_DOUBLE_EQ(model.a(), 0);
    EXPECT_DOUBLE_EQ(model.b(), 0);
    model.ab = {1, 2};
    EXPECT_DOUBLE_EQ(model.a(), 1);
    EXPECT_DOUBLE_EQ(model.b(), 2);
}

TEST(TestFrame, TestAffineTargetHost)
{
    AffineModel target{1, 2};
    AffineModel host{3, 4};

    // TODO : implement operator
}

TEST(TestErrorState, TestErrorStateFunctions)
{
    using Vector10d = Eigen::Matrix<double, Dim::kFrame, 1>;
    using Vector10dCRef = Eigen::Ref<const Vector10d>;

    Vector10d delta = Vector10d::Random();
    Sophus::SE3d T_w_cl = Sophus::SE3d::exp(delta.head<6>());

    ErrorState es{delta};

    EXPECT_DOUBLE_EQ(es.vec()[0], delta[0]);
    EXPECT_DOUBLE_EQ(es.ab_l()[0], delta[6]);
    EXPECT_DOUBLE_EQ(es.ab_l()[1], delta[7]);
    EXPECT_DOUBLE_EQ(es.ab_r()[0], delta[8]);
    EXPECT_DOUBLE_EQ(es.ab_r()[1], delta[9]);

    // EXPECT_DOUBLE_EQ(es.dT(), T_w_cl);

    ErrorState es2{delta};

    es2 += delta;

    EXPECT_DOUBLE_EQ(es2.vec()[0], 2 * delta[0]);
    EXPECT_DOUBLE_EQ(es2.ab_l()[0], 2 * delta[6]);
    EXPECT_DOUBLE_EQ(es2.ab_l()[1], 2 * delta[7]);
    EXPECT_DOUBLE_EQ(es2.ab_r()[0], 2 * delta[8]);
    EXPECT_DOUBLE_EQ(es2.ab_r()[1], 2 * delta[9]);
}


} // namespace adso