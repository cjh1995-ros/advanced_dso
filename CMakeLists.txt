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
)

# set lib for module test
add_library(image_reader src/image_reader.cpp)
target_link_libraries(image_reader ${INCLUDE_LIBRARIES})
add_library(point src/point.cpp)
target_link_libraries(point ${INCLUDE_LIBRARIES})
add_library(selector src/select.cpp)
target_link_libraries(selector 
    TBB::tbb
    absl::span
    ${INCLUDE_LIBRARIES}
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

set(TEST_SOURCE_FILES
    test/test_image_reader.cpp
    test/test_pixel_operate.cpp
    test/test_response_model.cpp
    test/test_selector.cpp
    test/test_vignette_model.cpp)

set(BENCHMARK_SOURCE_FILES
    benchmark/bm_pixel_operate.cpp)

add_executable(test_and_bm src/test_and_bm.cpp
    ${TEST_SOURCE_FILES}
    ${BENCHMARK_SOURCE_FILES})
target_link_libraries( test_and_bm
    ${INCLUDE_LIBRARIES}
    selector
    point
    gtest_main
    image_reader
    benchmark::benchmark
)

# set test sources
# add_executable(
#     test_image_reader 
#     test/test_image_reader.cpp
# )
# target_link_libraries(
#     test_image_reader
#     ${INCLUDE_LIBRARIES}
#     # GTest::gtest_main
#     gtest_main
#     image_reader
# )

# set test pixel
# add_executable(
#     test_pixel_operate
#     test/test_pixel_operate.cpp
# )
# target_link_libraries(
#     test_pixel_operate
#     ${INCLUDE_LIBRARIES}
#     # GTest::gtest_main
#     gtest_main
# )

# set test vignette
# add_executable(
#     test_vignette
#     test/test_vignette_model.cpp
# )
# target_link_libraries(
#     test_vignette
#     ${INCLUDE_LIBRARIES}
#     gtest_main
# )

# set test vignette
# add_executable(
#     test_response
#     test/test_response_model.cpp
# )
# target_link_libraries(
#     test_response
#     ${INCLUDE_LIBRARIES}
#     gtest_main
#     absl::span
# )

# set test selector
# add_executable(
#     test_selector
#     test/test_selector.cpp
# )
# target_link_libraries(
#     test_selector
#     ${INCLUDE_LIBRARIES}
#     gtest_main
#     selector
# )

############ Benchmark ############
# set benchmark sources
# add_executable(
#     bm_pixel_operate
#     benchmark/bm_pixel_operate.cpp
# )
# target_link_libraries(
#     bm_pixel_operate
#     ${INCLUDE_LIBRARIES}
#     benchmark::benchmark
#     benchmark::benchmark_main
# )

include(GoogleTest)
# gtest_discover_tests(test_image_reader)