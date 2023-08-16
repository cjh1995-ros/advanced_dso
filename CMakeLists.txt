cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyProject)

# Add C++ 11 Standard support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(
    thirdparty/googletest
    )

find_package(OpenCV REQUIRED PATHS /opt/homebrew/Cellar/opencv/4.8.0_1)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(Boost REQUIRED)

# # Include directories
# include_directories( 
#     include/module
#     ${OpenCV_INCLUDE_DIRS} 
#     )

# link_directories(${Eigen_INCLUDE_DIRS})

# set(INCLUDE_LIBRARIES
#     ${OpenCV_LIBS} 
#     Eigen3::Eigen
#     gtest
#     gtest_main)


# add_executable(test_pixel 
#     module_test/TestPixel.cpp)
# target_link_libraries(test_pixel 
#     ${INCLUDE_LIBRARIES}
#     )



# # Add module tester
# add_executable(test_image_reader 
#     module_test/TestImageReader.cpp
#     module/Image/Frame.cpp
#     module/Util/ImageReader.cpp)
# target_link_libraries(test_image_reader 
#     ${INCLUDE_LIBRARIES}
#     )