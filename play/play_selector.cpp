#include "util/viz.hpp"
#include "select.hpp"
#include "image_reader.hpp"
#include "image.hpp"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>

using std::cout, std::endl;

/**
 * @brief 
 * 
 */
ABSL_FLAG(std::string, 
          data_path, 
        //   "/Users/cjh/CJH_study_ws/advanced_dso/data/dataset-corridor4_512_16/dso/cam0/images",
          "/Users/cjh/CJH_study_ws/advanced_dso/data/data",
          "Path to the dataset directory.");
ABSL_FLAG(int32_t, cell_size, 16, "cell size");
ABSL_FLAG(int32_t, sel_level, 1, "select level");
ABSL_FLAG(int32_t, min_grad, 8, "minimum gradient");
ABSL_FLAG(int32_t, max_grad, 64, "maximum gradient");
ABSL_FLAG(double, min_ratio, 0.0, "minimum ratio");
ABSL_FLAG(double, max_ratio, 1.0, "maximum ratio");
ABSL_FLAG(bool, reselect, true, "reselect if ratio too low");


namespace adso
{

void Run()
{
    ImageReader reader(absl::GetFlag(FLAGS_data_path));
    
    SelectCfg cfg;
    cfg.set_vel = absl::GetFlag(FLAGS_sel_level);
    cfg.cell_size = absl::GetFlag(FLAGS_cell_size);
    cfg.min_grad = absl::GetFlag(FLAGS_min_grad);
    cfg.max_grad = absl::GetFlag(FLAGS_max_grad);
    cfg.min_ratio = absl::GetFlag(FLAGS_min_ratio);
    cfg.max_ratio = absl::GetFlag(FLAGS_max_ratio);
    cfg.reselect = absl::GetFlag(FLAGS_reselect);

    cout << "Data path: " << absl::GetFlag(FLAGS_data_path) << "\n";
    cout << "Set_vel: " << cfg.set_vel <<"\n";
    cout << "Cell Size: " << cfg.cell_size <<"\n";
    cout << "Min Gradient: " << cfg.min_grad <<"\n";
    cout << "Max Gradient: " << cfg.max_grad <<"\n";
    cout << "Min Ratio: " << cfg.min_ratio <<"\n";
    cout << "Max Ratio: " << cfg.max_ratio <<"\n";
    cout << "Do reselect?: " << cfg.reselect <<"\n";

    PixelSelector selector(cfg);

    int n = reader.getNumImages();
    int i = 0;

    cout << "Number of images: " << n << "\n";

    while (i < n)
    {
        cv::Mat img = reader.readImage(i);

        // Make image pyramid
        ImagePyramid grays;

        MakeImagePyramid(img, 4, grays);

        selector.Select(grays, 1);

        PixelGrid pixels = selector.pixels();
        
        cv::Scalar color(0, 0, 255); // red
        DrawSelectedPixels(img, pixels, color, 1);

        cv::imshow("vis", img);
        cv::waitKey(0);

        ++i;
    }
}

}

int main()
{
    adso::Run();
    return 0;
}