#include <benchmark/benchmark.h>
#include "select.hpp"
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>


namespace adso
{

namespace bm = benchmark;

constexpr int kNumLevels = 2;
constexpr int kImageSize = 640;
constexpr int kCellSize = 16;
const cv::Size kGridSize = {kImageSize / kCellSize, kImageSize / kCellSize};

/// ============================================================================
void BM_SelectLevel0(bm::State& state) {
    ImagePyramid images;
    MakeImagePyramid(MakeRandMat8U(kImageSize), kNumLevels, images);

    SelectCfg cfg;
    cfg.set_vel = 0;
    cfg.max_grad = 256;
    PixelSelector det{cfg};

    const auto gsize = static_cast<int>(state.range(0));
    for (auto _ : state) 
    {
        const auto n = det.Select(images, gsize);
        bm::DoNotOptimize(n);
    }
}
BENCHMARK(BM_SelectLevel0)->Arg(0)->Arg(1);

void BM_SelectLevel1(bm::State& state) {
    ImagePyramid images;
    MakeImagePyramid(MakeRandMat8U(kImageSize), kNumLevels, images);

    SelectCfg cfg;
    cfg.max_grad = 256;
    PixelSelector det{cfg};

    const auto gsize = static_cast<int>(state.range(0));
    for (auto _ : state) {
        const auto n = det.Select(images, gsize);
        bm::DoNotOptimize(n);
    }
}
BENCHMARK(BM_SelectLevel1)->Arg(0)->Arg(1);
}