#include "Util/ImageReader.hpp"


namespace adso 
{


ImageReader::ImageReader(std::string image_folder,
                         cv::Size image_size)
{
    this->m_img_new_size = image_size;
    this->getDir(image_folder);

    // check do we resize the image
    cv::Mat img = cv::imread(this->m_files[0]);
    if (img.size() != image_size)
        isResize = true;

    std::cout << "ImageReader: " << m_files.size() << " images found." << std::endl;
}   


cv::Mat ImageReader::readImage(int index)
{
    cv::Mat img = cv::imread(this->m_files[index]);

    if (img.empty())
    {
        std::cout << "ImageReader: Error reading image " << this->m_files[index] << std::endl;
        exit(1);
    }
    if (isResize)
        cv::resize(img, img, m_img_new_size);
    
    return img;
}

int ImageReader::getDir(const std::string& dir)
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
        this->m_files.push_back(entry.path());
    }

    std::sort(this->m_files.begin(), this->m_files.end());
    
    return (int)this->m_files.size();
}

} // namespace adso