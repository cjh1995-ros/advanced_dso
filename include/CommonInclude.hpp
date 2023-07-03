#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
 * Standard library includes
 */
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * Eigen includes
 */
// #include <Eigen/Core>
// #include <Eigen/Geometry>
// #include <Eigen/Dense>

// #include <Sophus/se3.hpp>
// #include <Sophus/so3.hpp>

// #define Eigen::Vec3f vec3f
// #define Eigen::Vec2f vec2f
// #define cv::Mat cMat

#define ADSO_NAMESPACE_BEGIN    namespace adso {
#define ADSO_NAMESPACE_END      }