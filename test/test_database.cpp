#include <gtest/gtest.h>

#include "image_reader.hpp"
#include "database.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <iostream>

using std::cout, std::endl, std::string;


namespace adso
{

TEST(TestDatabase, TestAddFrame)
{
    string data_path = "/Users/cjh/CJH_study_ws/advanced_dso/data/data";
    ImageReader reader(data_path);

    DatabaseCfg cfg;

    std::unique_ptr<Database> ptr_database_ = std::make_unique<Database>(cfg);
    
    int n = reader.getNumImages();
    int i = 0;

    while (i < n)
    {
        cv::Mat img = reader.readImage(i);

        ptr_database_->AddFrame(img);

        ++i;
    }

    EXPECT_EQ(n, ptr_database_->get_n_frames());
}

} // namespace adso