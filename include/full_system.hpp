#pragma once
#include <memory>
#include <database.hpp>

namespace adso
{

struct FullSystemCfg
{
    DatabaseCfg db_cfg;
};



class FullSystem
{
public:
    FullSystem() = default;
    explicit FullSystem(const FullSystemCfg& cfg)
    {
        ptr_database_ = std::make_unique<Database>(cfg.db_cfg);
    }

    ~FullSystem();

    bool AddActiveFrame(const cv::Mat image_l, const cv::Mat image_r, const double timestamp)
    {
        // TODO : Add active frame vo estimator -> estimate pose

        // TODO : Add if it is keyframe or not

        // TODO : if it is keyframe, add keyframe to database -> AddKeyFrame()

        // TODO : Add frame to database

    }

    bool AddFrame2Database(const cv::Mat image_l, const cv::Mat image_r, const double timestamp)
    {
        ptr_database_

    }

private:
    std::unique_ptr<Database> ptr_database_;
};



} // namespace adso