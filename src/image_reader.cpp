#include "image_reader.hpp"

namespace fs = std::filesystem;

namespace adso 
{

ImageReader::ImageReader(const std::string &path, cv::Size new_size)
{
    // set new size
    new_size_ = new_size;
    path_ = path;
    // push files into container
    fs::path p(path_);
    if (!fs::exists(p) || !fs::is_directory(p))
    {
        std::cerr << "Path " << path_ << " does not exist." << std::endl;
        exit(1);
    }

    fs::directory_iterator itr(p);
    for (const auto &p : fs::directory_iterator(p))
        if (p.path().extension() == ".jpg")
        {
            files_.push_back(p.path().string());
        }

    std::sort(files_.begin(), files_.end());
}

cv::Mat ImageReader::readImage(int idx)
{
    if (idx < 0 || idx >= files_.size())
    {
        std::cerr << "Index " << idx << " out of range." << std::endl;
        exit(1);
    }

    cv::Mat img = cv::imread(files_[idx], cv::IMREAD_GRAYSCALE);
    if (new_size_.width > 0 && new_size_.height > 0)
        cv::resize(img, img, new_size_);
    return img;
}


std::string ImageReader::logging() const
{
    return fmt::format("ImageReader: {} images in folder {}.", files_.size(), path_);
}




} // namespace adso