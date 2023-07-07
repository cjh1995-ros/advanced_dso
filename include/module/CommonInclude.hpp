#pragma once


/// @brief C++ includes
#include <iostream>
#include <vector>
#include <filesystem>
#include <random>
#include <cmath>
#include <chrono>

namespace fs = std::filesystem;

/// @brief OpenCV Includes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>


/// @brief Eigen Includes
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::Matrix<float, 8, 1> Vector8f;
typedef Eigen::Matrix<float, 8, 8> Matrix88f;

/// @brief Sophus Includes
// #include <Sophus/se3.hpp>
// #include <Sophus/so3.hpp>

