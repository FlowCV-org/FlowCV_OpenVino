//
// FlowCV Plugin -
// OpenVino Pose Estimation
//

#ifndef FLOWCV_PLUGIN_TEST_HPP_
#define FLOWCV_PLUGIN_TEST_HPP_
#include <DSPatch.h>
#include "FlowCV_Types.hpp"
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <string>

#include <monitors/presenter.h>

#include <utils/args_helper.hpp>
#include <utils/images_capture.h>
#include <utils/performance_metrics.hpp>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>
#include <utils/default_flags.hpp>
#include <unordered_map>
#include <gflags/gflags.h>

#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>

#include <models/hpe_model_associative_embedding.h>
#include <models/hpe_model_openpose.h>

namespace DSPatch::DSPatchables
{
namespace internal
{
class PoseEstimation2D;
}

class DLLEXPORT PoseEstimation2D final : public Component
{
public:
    PoseEstimation2D();
    void UpdateGui(void *context, int interface) override;
    bool HasGui(int interface) override;
    std::string GetState() override;
    void SetState(std::string &&json_serialized) override;

protected:
    void Process_( SignalBus const& inputs, SignalBus& outputs ) override;
    bool InitModel(const std::string& networkType, int width, int height);

private:
    std::unique_ptr<internal::PoseEstimation2D> p;
    std::mutex io_mutex_;
    cv::Mat last_frame_;
    InferenceEngine::Core core_;
    bool init_once_;
    bool first_proc_;
    bool init_success_;
    bool is_async_mode_;
    bool is_mode_changed_;
    int network_type_;
    int precision_mode_;
    int model_num_;
    int device_index_;
    std::vector<std::string> devices_;
    std::unique_ptr<OutputTransform> output_transform_;
    cv::Size output_resolution_;
    std::chrono::steady_clock::time_point start_time_;
    int64_t frame_num_;
    std::unique_ptr<AsyncPipeline> pipeline_;
    std::unique_ptr<ResultBase> result_;
    std::vector<HumanPose> poses_;
};

EXPORT_PLUGIN( PoseEstimation2D )

}  // namespace DSPatch::DSPatchables

#endif //FLOWCV_PLUGIN_TEST_HPP_
