#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace adso
{


/// @brief Simplified BlockedRange, gsize <= 0 will only use single thread
struct BlockedRange
{
    int begin_{};
    int end_{};
    int grain_size_{};


    BlockedRange() = default;
    BlockedRange(int begin, int end, int grain_size):
        begin_(begin), end_(end), grain_size_(grain_size <= 0 ? end - begin : grain_size) {}

    auto ToTbb() const noexcept
    {
        return tbb::blocked_range<int>(begin_, end_, grain_size_);
    }
};

/// @brief Wrapper for tbb::parallel_reduce
template <typename T, typename Func, typename Reduc>
T ParallelReduce(const BlockedRange& range,
                 const T& identity,
                 const Func& func,
                 const Reduc& reduction)
{
    return tbb::parallel_reduce(
        range.ToTbb(), 
        identity,
        [&](const auto& block, T local)
        {
            for (int i = block.begin(); i < block.end(); ++i)
                func(i, local);

            return local;
        },
        reduction
    );
}

/// @brief Wrapper for tbb::parallel_for
template <typename Func>
void ParallelFor(const BlockedRange& range,
                 const Func& func)
{
    tbb::parallel_for(
        range.ToTbb(),
        [&](const auto& block)
        {
            for (int i = block.begin(); i < block.end(); ++i)
                func(i);
        }
    );
}

}  // namespace adso