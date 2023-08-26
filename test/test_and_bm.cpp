#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
    // Initialize and run gtest
    ::testing::InitGoogleTest(&argc, argv);
    int gtest_result = RUN_ALL_TESTS();

    // Initialize and run gbenchmark
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    return gtest_result;
}
