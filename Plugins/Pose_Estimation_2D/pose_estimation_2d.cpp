//
// FlowCV Plugin -
// OpenVino Pose Estimation
//
#include "pose_estimation_2d.hpp"
#include "imgui.h"
#include "imgui_instance_helper.hpp"

#define _X86_
#include <filesystem>
#include <memory>
#ifdef _WIN32
#include <minwindef.h>
#include "libloaderapi.h"
#else
#include <libgen.h>
#include <unistd.h>
#include <dlfcn.h>
#endif

using namespace DSPatch;
using namespace DSPatchables;

int32_t global_inst_counter = 0;

namespace DSPatch::DSPatchables::internal
{
class PoseEstimation2D
{
};
}  // namespace DSPatch

PoseEstimation2D::PoseEstimation2D()
    : Component( ProcessOrder::OutOfOrder )
    , p( new internal::PoseEstimation2D() )
{
    // Name and Category
    SetComponentName_("Pose_Estimation_2D");
    SetComponentCategory_(Category::Category_OpenVino);
    SetComponentAuthor_("Richard");
    SetComponentVersion_("0.1.0");
    SetInstanceCount(global_inst_counter);
    global_inst_counter++;

    // 1 inputs
    SetInputCount_( 1, {"in"}, {IoType::Io_Type_CvMat} );

    // 1 outputs
    SetOutputCount_( 2, {"vis", "poses"}, {IoType::Io_Type_CvMat, IoType::Io_Type_JSON} );

    is_async_mode_ = false;
    init_once_ = true;
    first_proc_ = true;
    frame_num_ = 0;
    model_num_ = 0;
    is_mode_changed_ = false;
    init_success_ = false;
    network_type_ = 0;
    precision_mode_ = 2;
    device_index_ = 0;

    devices_ = core_.GetAvailableDevices();
    for (int i = 0; i < devices_.size(); i++) {
        if (devices_.at(i) == "GNA")
            devices_.erase(devices_.begin() + i);
    }

    SetEnabled(true);

}

#ifndef _WIN32
static const char *my_fname(void) {
    Dl_info dl_info;
    dladdr((void*)my_fname, &dl_info);
    return(dl_info.dli_fname);
}
#endif

bool PoseEstimation2D::InitModel(const std::string& networkType, int width, int height)
{
    std::string modelPath;

#ifdef _WIN32
    char path[255];
    HMODULE hm = nullptr;

    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          (LPCSTR)"Pose_Estimation_2D.fp", &hm);
    GetModuleFileNameA(hm, path, sizeof(path));
    std::filesystem::path filePath = path;
    std::string pluginPath = filePath.remove_filename().string();
#else
    std::filesystem::path filePath = my_fname();
    std::string pluginPath = filePath.remove_filename().string();
#endif

    int mn = 1;
    if (network_type_ > 0)
        mn = model_num_ + 2;
    int pm = precision_mode_;
    if (model_num_ != 0 && pm == 0)
        pm = 1;

    modelPath = pluginPath;
    modelPath += "human-pose-estimation-000";
    modelPath += std::to_string(mn);
    modelPath += std::filesystem::path::preferred_separator;

    if (pm == 0)
        modelPath += "FP16-INT8";
    else if (pm == 1)
        modelPath += "FP16";
    else if (pm == 2)
        modelPath += "FP32";

    modelPath += std::filesystem::path::preferred_separator;
    modelPath += "human-pose-estimation-000";
    modelPath += std::to_string(mn);
    modelPath += ".xml";

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "Pose Estimation Model File is Missing - " << modelPath << std::endl;
        return false;
    }

    std::unique_ptr<ModelBase> model;
    double aspectRatio = width / static_cast<double>(height);

    if (networkType == "openpose") {
        model = std::make_unique<HPEOpenPose>(modelPath, aspectRatio, 0, 0.1f);
    }
    else if (networkType == "ae") {
        model = std::make_unique<HpeAssociativeEmbedding>(modelPath, aspectRatio, 0, 0.1f);
    }
    pipeline_.reset();
    pipeline_ = std::make_unique<AsyncPipeline>(std::move(model),
                                                ConfigFactory::getMinLatencyConfig(devices_.at(device_index_), "", "", false, 0),
                                                core_);
    return true;
}

void PoseEstimation2D::Process_( SignalBus const& inputs, SignalBus& outputs )
{
    static const cv::Scalar colors[HPEOpenPose::keypointsNumber] = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
        cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
    };
    static const std::pair<int, int> keypointsOP[] = {
        {1, 2}, {1, 5}, {2, 3}, {3, 4},  {5, 6}, {6, 7},
        {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13},
        {1, 0}, {0, 14},{14, 16}, {0, 15}, {15, 17}
    };
    static const std::pair<int, int> keypointsAE[] = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11},
        {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10},
        {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}
    };
    // OpenVino processing is already multithreaded so no need to multithread this
    if (io_mutex_.try_lock()) { // Try lock so other threads will skip if locked instead of waiting
        cv::Mat frame;
        // Handle Input Code Here
        auto in1 = inputs.GetValue<cv::Mat>(0);
        if (!in1) {
            io_mutex_.unlock();
            return;
        }

        // Handle Output Code Here
        if (!in1->empty()) {
            in1->copyTo(frame);

            if (!last_frame_.empty()) {
                if (frame.size != last_frame_.size) {
                    init_once_ = true;
                    first_proc_ = true;
                }
            }

            if (init_once_) {
                init_success_ = false;
                if (network_type_ == 0)
                    init_success_ = InitModel("openpose", frame.cols, frame.rows);
                else if (network_type_ == 1)
                    init_success_ = InitModel("ae", frame.cols, frame.rows);
                init_once_ = false;
            }

            if (init_success_) {
                if (first_proc_) {
                    output_transform_.reset();
                    frame.copyTo(last_frame_);
                    output_transform_ = std::make_unique<OutputTransform>(frame.size(), frame.size());
                    output_resolution_ = output_transform_->computeResolution();
                    start_time_ = std::chrono::steady_clock::now();
                    frame_num_ = pipeline_->submitData(ImageInputData(frame),
                                                       std::make_shared<ImageMetaData>(frame, start_time_));
                    first_proc_ = false;
                    outputs.SetValue(0, frame);
                    io_mutex_.unlock();
                    return;
                }

                if (pipeline_->isReadyToProcess()) {
                    //--- Capturing frame
                    start_time_ = std::chrono::steady_clock::now();
                    frame_num_ = pipeline_->submitData(ImageInputData(frame),
                                                       std::make_shared<ImageMetaData>(frame, start_time_));
                }

                pipeline_->waitForData();

                result_ = pipeline_->getResult();
                if (result_) {
                    poses_ = result_->asRef<HumanPoseResult>().poses;
                    if (!poses_.empty()) {
                        nlohmann::json json_out;
                        nlohmann::json jPoses;
                        for (const auto &pose: poses_) {
                            // Just grabbing the keypoints from the first pose for now
                            std::vector<float> keypoints;
                            nlohmann::json jPose;
                            for (auto &key: pose.keypoints) {
                                keypoints.emplace_back(key.x);
                                keypoints.emplace_back(key.y);
                            }
                            jPose["keypoints"] = keypoints;
                            jPose["key_count"] = keypoints.size() / 2;
                            jPose["score"] = pose.score;
                            jPoses.emplace_back(jPose);
                        }
                        json_out["data_type"] = "poses";
                        json_out["data"] = jPoses;
                        nlohmann::json ref;
                        ref["w"] = frame.cols;
                        ref["h"] = frame.rows;
                        json_out["ref_frame"] = ref;
                        outputs.SetValue(1, json_out);
                    }
                }
                frame.copyTo(last_frame_);

                const int stickWidth = 4;
                const cv::Point2f absentKeypoint(-1.0f, -1.0f);
                for (auto &pose: poses_) {
                    for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++) {
                        if (pose.keypoints[keypointIdx] != absentKeypoint) {
                            output_transform_->scaleCoord(pose.keypoints[keypointIdx]);
                            cv::circle(frame, pose.keypoints[keypointIdx], 4, colors[keypointIdx], -1);
                        }
                    }
                }
                std::vector<std::pair<int, int>> limbKeypointsIds;
                if (!poses_.empty()) {
                    if (poses_[0].keypoints.size() == HPEOpenPose::keypointsNumber) {
                        limbKeypointsIds.insert(limbKeypointsIds.begin(), std::begin(keypointsOP), std::end(keypointsOP));
                    } else {
                        limbKeypointsIds.insert(limbKeypointsIds.begin(), std::begin(keypointsAE), std::end(keypointsAE));
                    }
                }
                cv::Mat pane = frame.clone();
                for (auto pose: poses_) {
                    for (const auto &limbKeypointsId: limbKeypointsIds) {
                        std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keypoints[limbKeypointsId.first],
                                                                          pose.keypoints[limbKeypointsId.second]);
                        if (limbKeypoints.first == absentKeypoint
                            || limbKeypoints.second == absentKeypoint) {
                            continue;
                        }

                        float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
                        float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
                        cv::Point difference = limbKeypoints.first - limbKeypoints.second;
                        double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
                        int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
                        std::vector<cv::Point> polygon;
                        cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stickWidth),
                                         angle, 0, 360, 1, polygon);
                        cv::fillConvexPoly(pane, polygon, colors[limbKeypointsId.second]);
                    }
                }
                cv::addWeighted(frame, 0.4, pane, 0.6, 0, frame);

                outputs.SetValue(0, frame);
            }
        }
        io_mutex_.unlock();
    }
}

bool PoseEstimation2D::HasGui(int interface)
{
    if (interface == (int)FlowCV::GuiInterfaceType_Controls) {
        return true;
    }

    return false;
}

void PoseEstimation2D::UpdateGui(void *context, int interface)
{
    auto *imCurContext = (ImGuiContext *)context;
    ImGui::SetCurrentContext(imCurContext);

    if (interface == (int)FlowCV::GuiInterfaceType_Controls) {
        ImGui::SetNextItemWidth(120);
        if (ImGui::Combo(CreateControlString("HW Accel Device", GetInstanceName()).c_str(), &device_index_, [](void* data, int idx, const char** out_text) {
            *out_text = ((const std::vector<std::string>*)data)->at(idx).c_str();
            return true;
        }, (void*)&devices_, (int)devices_.size())) {
            init_once_ = true;
            first_proc_ = true;
        }
        if (ImGui::Combo(CreateControlString("Pose Network Type", GetInstanceName()).c_str(), &network_type_, "Openpose (MobileNet v1)\0AE (EfficientHRNet)\0\0")) {
            init_once_ = true;
            first_proc_ = true;
        }
        if (ImGui::Combo(CreateControlString("Precision", GetInstanceName()).c_str(), &precision_mode_, "INT8\0FP16\0FP32\0\0")) {
            init_once_ = true;
            first_proc_ = true;
        }
        if (network_type_ > 0) {
            if (ImGui::Combo(CreateControlString("Model Number", GetInstanceName()).c_str(), &model_num_, " 1\0 2\0 3\0 4\0 5\0 6\0\0")) {
                init_once_ = true;
                first_proc_ = true;
            }
        }
    }

}

std::string PoseEstimation2D::GetState()
{
    using namespace nlohmann;

    json state;

    state["network_type"] = network_type_;
    state["precision_mode"] = precision_mode_;
    state["model_num"] = model_num_;
    state["device_idx"] = device_index_;
    state["device"] = devices_.at(device_index_);

    std::string stateSerialized = state.dump(4);

    return stateSerialized;
}

void PoseEstimation2D::SetState(std::string &&json_serialized)
{
    using namespace nlohmann;

    json state = json::parse(json_serialized);

    if (state.contains("network_type"))
        network_type_ = state["network_type"].get<int>();
    if (state.contains("precision_mode"))
        precision_mode_ = state["precision_mode"].get<int>();
    if (state.contains("model_num"))
        model_num_ = state["model_num"].get<int>();
    if (state.contains("device_idx"))
        device_index_ = state["device_idx"].get<int>();
    if (state.contains("device")) {
        auto devStr = state["device"].get<std::string>();
        if (device_index_ > devices_.size() - 1)
            device_index_ = 0;
        if (devStr != devices_.at(device_index_)) {
            bool foundMatch = false;
            for (int i = 0; i < devices_.size(); i++) {
                if (devices_.at(i) == devStr) {
                    device_index_ = i;
                    foundMatch = true;
                    break;
                }
            }
            if (!foundMatch)
                device_index_ = 0;

            init_once_ = true;
            first_proc_ = true;
        }
    }
}
