//
// FlowCV Plugin -
// OpenVino Head, Face, Emotion Detection
//

#ifndef FLOWCV_PLUGIN_HEAD_FACE_EMOTION_HPP_
#define FLOWCV_PLUGIN_HEAD_FACE_EMOTION_HPP_
#include <DSPatch.h>
#include "FlowCV_Types.hpp"
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <list>
#include <set>
#include <fstream>
#include <random>
#include <memory>

#include <gflags/gflags.h>
#include <inference_engine.hpp>
#include <monitors/presenter.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"

namespace DSPatch::DSPatchables
{
namespace internal
{
class HeadFaceEmotion;
}

class DLLEXPORT HeadFaceEmotion final : public Component
{
  public:
    HeadFaceEmotion();
    void UpdateGui(void *context, int interface) override;
    bool HasGui(int interface) override;
    std::string GetState() override;
    void SetState(std::string &&json_serialized) override;

  protected:
    void Process_( SignalBus const& inputs, SignalBus& outputs ) override;
    bool InitModels();
    int MatchDeviceIndex(std::string &devStr, int devIdx);

  private:
    std::unique_ptr<internal::HeadFaceEmotion> p;
    std::mutex io_mutex_;
    cv::Mat last_frame_;
    InferenceEngine::Core core_;
    std::unique_ptr<FaceDetection> face_detector_;
    std::unique_ptr<HeadPoseDetection> head_pose_detector_;
    std::unique_ptr<FacialLandmarksDetection> facial_landmarks_detector_;
    std::unique_ptr<EmotionsDetection> emotion_detector_;
    std::unique_ptr<Visualizer> visualizer_;
    std::list<Face::Ptr> faces_;
    std::vector<std::string> devices_;
    int face_detect_device_;
    int face_max_batch_;
    float face_threshold_;
    int face_precision_;
    int head_pose_device_;
    int head_max_batch_;
    int head_precision_;
    int landmark_device_;
    int landmark_max_batch_;
    int landmark_precision_;
    int emotion_device_;
    int emotion_max_batch_;
    int emotion_precision_;
    size_t faceId{};
    bool init_success_{};
    bool init_once_{};
    bool firstProc{};
    bool head_pose_enabled_;
    bool emotion_enabled_;
    bool face_landmarks_enabled_;
};

EXPORT_PLUGIN( HeadFaceEmotion )

}  // namespace DSPatch::DSPatchables

#endif //FLOWCV_PLUGIN_HEAD_FACE_EMOTION_HPP_
