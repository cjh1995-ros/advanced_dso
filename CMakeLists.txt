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

# set dependencies
include_directories( 
    include
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

# main source
add_executable(adso src/main.cpp)
target_link_libraries(
    adso
    ${INCLUDE_LIBRARIES}
)


# Test sources
# for gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# set test sources
add_executable(
    test_image_reader 
    test/test_image_reader.cpp
)
target_link_libraries(
    test_image_reader
    ${INCLUDE_LIBRARIES}
    GTest::gtest_main
    image_reader
)

include(GoogleTest)
gtest_discover_tests(test_image_reader)