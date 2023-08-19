#pragma once
#include <opencv2/core.hpp>
#include "keyframe.hpp"
#include "vignette_model.hpp"
#include "response_model.hpp"

#include <vector>


namespace adso
{

class Database
{
public:
    Database() = default;
    explicit Database(const cv::Size& size);

    //////////////// Handling keyframe ///////////////////////
    /// @brief add keyframe to database
    void AddKeyFrame(const KeyFrame& frame) { frames_.push_back(frame); }

    /// @brief get keyframe from database
    const KeyFrame& GetKeyFrame(int idx) const { return frames_[idx]; }

    /// @brief remove keyframe from database
    void RemoveKeyFrame(int idx) { frames_.erase(frames_.begin() + idx); }

    /// @brief remove last frame from database
    void RemoveOldFrame() { frames_.pop_front(); }

    //////////////// Handling Vignette ///////////////////////
    void SetVignetteModel(const VignetteModel& vig) { vignette_model_ = vig; }
    const VignetteModel& GetVignetteModel() const { return vignette_model_; }

    //////////////// Handling Response ///////////////////////
    void SetResponseModel(const ResponseModel& res) { response_model_ = res; }
    const ResponseModel& GetResponseModel() const { return response_model_; }


    //////////////// Handling Graph ///////////////////////
    void AddEdge(int idx1, int idx2) { graph_[idx1].push_back(idx2); }
    void RemoveEdge(int idx1, int idx2) { graph_[idx1].erase(graph_[idx1].begin() + idx2); }
    // void RemoveNode(int idx) 
    // {
    //     // remove all edges to from this node
    //     graph_.erase(graph_.begin() + idx); 
    //     for (auto& edges : graph_)
    //     {
    //         edges.erase(edges.begin() + idx);
    //     }
    // }


    /////////////////// Visualize /////////////////////////////
    /**
     * @todo: 
     *  1. visualize tracking with feature points in last image
     *  2. visualize corrected photo from vignette, response and exposure
    */


private:
    cv::Size size_;
    std::vector<KeyFrame> frames_;
    VignetteModel vignette_model_;
    ResponseModel response_model_;

    std::vector<std::vector<int>> graph_; // graph of keyframe
};




} // namespace adso