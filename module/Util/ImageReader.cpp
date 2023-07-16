#include "Util/ImageReader.hpp"


namespace adso 
{
ImageReader::ImageReader(std::string image_folder)
{
    setImages(image_folder);

    // check do we resize the image
    cv::Mat img = cv::imread(files_[0]);
    img_new_size_ = img.size();
    is_resize_ = false;
}   


ImageReader::ImageReader(std::string image_folder,
                         cv::Size image_size)
{
    img_new_size_ = image_size;
    setImages(image_folder);

    // check do we resize the image
    cv::Mat img = cv::imread(files_[0]);
    if (img.size() != img_new_size_)
        is_resize_ = true;
        

    std::cout << "ImageReader: " << files_.size() << " images found." << std::endl;
}   


cv::Mat ImageReader::at(int index)
{
    cv::Mat img = cv::imread(files_[index]);

    if (img.empty())
    {
        std::cout << "ImageReader: Error reading image " << files_[index] << std::endl;
        exit(1);
    }
    if (is_resize_)
        cv::resize(img, img, img_new_size_);
    
    return img;
}

int ImageReader::setImages(const std::string& dir)
{
    fs::path path(dir);

    // not a directory?
    if(!fs::is_directory(path))
    {
        std::cout << "This is not a directory: " << dir << std::endl;
        return -1;
    }
    
    fs::directory_iterator itr(path);
    
    // add file names in the directory to the vector
    for (const fs::directory_entry& entry : fs::directory_iterator(path))
    {
        files_.push_back(entry.path());
    }

    std::sort(files_.begin(), files_.end());
    
    return (int)files_.size();
}

} // namespace adso