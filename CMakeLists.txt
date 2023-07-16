cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyProject)

# Add C++ 11 Standard support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(
    thirdparty/googletest
    )

find_package(OpenCV REQUIRED PATHS /opt/homebrew/Cellar/opencv/4.8.0)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories( 
    include/module
    ${OpenCV_INCLUDE_DIRS} 
    )

link_directories(${Eigen_INCLUDE_DIRS})

# Add module tester
add_executable(test_image_reader 
    module_test/TestImageReader.cpp)
target_link_libraries(test_image_reader 
    ${OpenCV_LIBS} 
    Eigen3::Eigen
    gtest
    gtest_main
    )


# add_executable(test_image_and_exposure 
#    module_test/TestImageAndExposure.cpp)


# target_link_libraries( test_image_and_exposure 
#     ${OpenCV_LIBS} 
#     ${Eigen_LIBRARIES})

add_executable(test_solve module_test/TestSolve.cpp)
target_link_libraries( test_solve
    ${OpenCV_LIBS} 
    Eigen3::Eigen)

add_executable(test_frame module_test/TestFrame.cpp)
target_link_libraries( test_frame
        ${OpenCV_LIBS} 
        Eigen3::Eigen
        gtest
        gtest_main)