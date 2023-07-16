#include "Image/Frame.hpp"

namespace adso
{

void Frame::genGradientImage()
{
    for(int i = 0; i < plvl_; i++)
    {
        cv::Mat img = img_pyramids_[i];
        cv::Mat img_grad = cv::Mat::zeros(img.size(), CV_32FC3);
        
        int wl = img.cols;
        int hl = img.rows;

        for (int y = 1; y < hl - 1; y++)
        for (int x = 1; x < wl - 1; x++)
        {
            float dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
            float dy = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
            img_grad.at<cv::Vec3f>(y, x)[0] = std::sqrtf(dx * dx + dy * dy);
            img_grad.at<cv::Vec3f>(y, x)[1] = dx;
            img_grad.at<cv::Vec3f>(y, x)[2] = dy;
        }
        
        sizes_.push_back(img.size());
        gradients_.push_back(img_grad);
    }
}
}