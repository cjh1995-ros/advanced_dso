#pragma once

#include "CommonInclude.hpp"


namespace adso 
{

/// @brief ImageReader class
/// @details This class reads images from a folder.
///          It can also read calibration file from a string.
class ImageReader
{
public:

    /// TODO: add reading calibration file from string
    /// @brief Construct a new ImageReader object
    /// @param image_folder 
    /// @param image_size
    ImageReader(std::string image_folder);
    ImageReader(std::string image_folder, cv::Size image_size);

    /// @brief Read image from the folder
    /// @param index 
    /// @return image at index 
    cv::Mat readImage(int index);

    /// @brief  Get the number of images in the folder
    /// @return 
    inline int getNumImages(){return static_cast<int>(m_files.size());};

    /// @brief Get the Dir object
    /// @param dir 
    /// @return 
    int getDir(const std::string& dir);

private:
    bool is_resize_ = false;
    cv::Size img_new_size_;
    std::vector<std::string> files_;
};



} // namespace adso