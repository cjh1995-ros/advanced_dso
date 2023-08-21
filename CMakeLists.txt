cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyProject)

# Add C++ 11 Standard support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# brew packages are in /opt/homebrew/opt
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt")

# find dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(Boost REQUIRED)
find_package(fmt REQUIRED)
find_package(absl REQUIRED)
find_package(TBB REQUIRED)

# set dependencies
include_directories( 
    include
    include/util
    ${OpenCV_INCLUDE_DIRS} 
)
link_directories(${Eigen_INCLUDE_DIRS})

# set include libraries
set(INCLUDE_LIBRARIES
    ${OpenCV_LIBS} 
    Eigen3::Eigen
    glog::glog
    fmt::fmt
    pthread
    TBB::tbb
    absl::span
    gtest
)

# install gtest and google-benchmark
include(FetchContent)
FetchContent_Declare(
  googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.0
)
FetchContent_Declare(
  googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.7.0

)
FetchContent_MakeAvailable(
    googletest
    googlebenchmark
)

enable_testing()

file(GLOB SOURCES "src/*.cpp")

set(TEST_SOURCE_FILES
    test/test_image_reader.cpp
    test/test_pixel_operate.cpp
    test/test_response_model.cpp
    test/test_selector.cpp
    test/test_vignette_model.cpp)

set(BENCHMARK_SOURCE_FILES
    benchmark/bm_pixel_operate.cpp
    benchmark/bm_select.cpp)

add_executable(test_and_bm src/test_and_bm.cpp
    ${TEST_SOURCE_FILES}
    ${BENCHMARK_SOURCE_FILES}
    ${SOURCES}
)
target_link_libraries( test_and_bm
    ${INCLUDE_LIBRARIES}
    gtest_main
    benchmark::benchmark
)

include(GoogleTest)
# gtest_discover_tests(test_and_bm)