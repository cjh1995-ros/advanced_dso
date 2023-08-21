#include <benchmark/benchmark.h>
#include <util/pixel_operate.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

namespace adso
{

constexpr int kSize = 320;
constexpr int kHalfSize = kSize / 2; // 160
const Eigen::Vector2d kUv = {kHalfSize, kHalfSize}; // 160, 160
const cv::Point2d kPx = {kHalfSize, kHalfSize}; // 160, 160
const cv::Mat kImage = cv::Mat::ones(kSize, kSize, CV_8UC1);

void BM_ValAtD(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        benchmark::DoNotOptimize(ValAtD<uchar>(kImage, kPx));
    }
}
BENCHMARK(BM_ValAtD);

void BM_ValAtE(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        benchmark::DoNotOptimize(ValAtE<uchar>(kImage, kUv));
    }
}
BENCHMARK(BM_ValAtE);

void BM_GradAtD(benchmark::State& state) 
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GradAtD<uchar>(kImage, kPx));
    }
}
BENCHMARK(BM_GradAtD);

void BM_GradAtE(benchmark::State& state) 
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GradAtE<uchar>(kImage, kUv));
    }
}
BENCHMARK(BM_GradAtE);


void BM_ValGradAtD(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(GradAtD<uchar>(kImage, kPx));
    }
}
BENCHMARK(BM_ValGradAtD);

void BM_ValGradAtE(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        benchmark::DoNotOptimize(GradValAtE<uchar>(kImage, kUv));
    }
}
BENCHMARK(BM_ValGradAtE);

void BM_ValGradAtDSep(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        benchmark::DoNotOptimize(ValAtD<uchar>(kImage, kPx));
        benchmark::DoNotOptimize(GradAtD<uchar>(kImage, kPx));
    }
}
BENCHMARK(BM_ValGradAtDSep);


void BM_GradAtI(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        benchmark::DoNotOptimize(GradAtI<uchar>(kImage, kPx));
    }
}
BENCHMARK(BM_GradAtI);


void BM_SobelAtI(benchmark::State& state) 
{
    for (auto _ : state) 
    {
        benchmark::DoNotOptimize(SobelAtI<uchar>(kImage, kPx));
    }
}
BENCHMARK(BM_SobelAtI);

}