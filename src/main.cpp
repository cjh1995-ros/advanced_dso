#include "CLI11.hpp"
#include "image_reader.hpp"
#include <string>
#include <iostream>

int main(int argc, char **argv) 
{
    CLI::App app("Advanced Direct Sparse Odometry");
    std::string image_folder;

    app.add_option("-i,--image-folder", image_folder, "Folder with image files to read.", true);

    CLI11_PARSE(app, argc, argv);

    adso::ImageReader reader(image_folder);

    for (int i = 0; i < reader.getNumImages(); ++i) 
    {
        cv::Mat img = reader.readImage(i);
        std::cout << reader.logging() << std::endl;
    }

    return 0;
}