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
    cv::Mat at(int index);

    /// @brief  Get the number of images in the folder
    /// @return 
    inline int getNumImages(){return static_cast<int>(files_.size());};

    /// @brief Get the Dir object
    /// @param dir 
    /// @return 
    int setImages(const std::string& dir);

private:
    bool is_resize_ = false;
    cv::Size img_new_size_;
    std::vector<std::string> files_;
};


class StereoImageReader
{
private:
    std::unique_ptr<ImageReader> image_reader_left_;
    std::unique_ptr<ImageReader> image_reader_right_;

    std::string folder_path_;

public:
    StereoImageReader(std::string folder_path): folder_path_(folder_path)
    {
        std::string left_folder = folder_path_ + "/cam0/images";
        std::string right_folder = folder_path_ + "/cam1/images";

        image_reader_left_ = std::make_unique<ImageReader>(left_folder);
        image_reader_right_ = std::make_unique<ImageReader>(right_folder);

        if (!isSameSize())
        {
            std::cout << "StereoImageReader: left and right images have different number of images." << std::endl;
            exit(1);
        }
    }

    std::vector<cv::Mat> at(int index)
    {
        std::vector<cv::Mat> images;
        images.push_back(image_reader_left_->at(index));
        images.push_back(image_reader_right_->at(index));
        return images;
    }

    bool isSameSize()
    {
        return image_reader_left_->getNumImages() == image_reader_right_->getNumImages();
    }

};

} // namespace adso