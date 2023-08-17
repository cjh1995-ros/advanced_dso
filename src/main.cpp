#include "CLI11.hpp"
#include <string>
#include <iostream>

int main(int argc, char **argv) 
{
    CLI::App app("Advanced Direct Sparse Odometry");
    std::string image_folder;

    app.add_option("-i,--image-folder", image_folder, "Folder with image files to read.", true);

    CLI11_PARSE(app, argc, argv);

    std::cout << "Image folder: " << image_folder << std::endl;

    return 0;
}