#pragma once

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <glog/logging.h>
#include <fmt/format.h>

namespace adso
{

class ImageReader
{
public:
    ImageReader(const std::string &path, cv::Size new_size = cv::Size(0, 0));
    cv::Mat readImage(int idx);

    int getNumImages() const { return files_.size(); }

    // log the image num and folder path
    std::string logging() const;

private:
    cv::Size new_size_;
    std::vector<std::string> files_;
    std::string path_;
};


} // namespace adso