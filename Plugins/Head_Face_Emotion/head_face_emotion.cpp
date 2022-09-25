//
// FlowCV Plugin -
// OpenVino Head, Face, Emotion Detection
//

#include "head_face_emotion.hpp"
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
class HeadFaceEmotion
{
};
}  // namespace DSPatch

HeadFaceEmotion::HeadFaceEmotion()
    : Component( ProcessOrder::OutOfOrder )
    , p( new internal::HeadFaceEmotion() )
{
    // Name and Category
    SetComponentName_("Head_Face_Emotion");
    SetComponentCategory_(Category::Category_OpenVino);
    SetComponentAuthor_("Richard");
    SetComponentVersion_("0.1.0");
    SetInstanceCount(global_inst_counter);
    global_inst_counter++;

    // 1 inputs
    SetInputCount_( 1, {"in"}, {IoType::Io_Type_CvMat} );

    // 1 outputs
    SetOutputCount_( 2, {"vis", "faces"}, {IoType::Io_Type_CvMat, IoType::Io_Type_JSON} );

    firstProc = true;
    init_once_ = true;
    init_success_ = false;
    head_pose_enabled_ = false;
    emotion_enabled_ = false;
    face_landmarks_enabled_ = false;
    face_detect_device_ = 0;
    face_max_batch_ = 16;
    face_threshold_ = 0.75f;
    face_precision_ = 2;
    head_pose_device_ = 0;
    head_max_batch_ = 16;
    head_precision_ = 2;
    landmark_device_ = 0;
    landmark_max_batch_ = 16;
    landmark_precision_ = 2;
    emotion_device_ = 0;
    emotion_max_batch_ = 16;
    emotion_precision_ = 2;

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

bool HeadFaceEmotion::InitModels()
{
    faceId = 0;
    std::string modelPath;

#ifdef _WIN32
    char path[255];
    HMODULE hm = nullptr;

    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)"Head_Face_Emotion.fp", &hm);
    GetModuleFileNameA(hm, path, sizeof(path));
    std::filesystem::path filePath = path;
    std::string pluginPath = filePath.remove_filename().string();
#else
    std::filesystem::path filePath = my_fname();
    std::string pluginPath = filePath.remove_filename().string();
#endif

    std::vector<std::string> preStr = {"FP16-INT8", "FP16", "FP32"};

    modelPath = pluginPath;

    // Face Model Path
    std::string faceDetectionPath = modelPath;
    faceDetectionPath += std::filesystem::path::preferred_separator;
    faceDetectionPath += "face-detection-retail-0005";
    faceDetectionPath += std::filesystem::path::preferred_separator;

    // Head Model Path
    std::string headPosePath = modelPath;
    headPosePath += std::filesystem::path::preferred_separator;
    headPosePath += "head-pose-estimation-adas-0001";
    headPosePath += std::filesystem::path::preferred_separator;

    // Landmark Model Path
    std::string faceLandmarkPath = modelPath;
    faceLandmarkPath += std::filesystem::path::preferred_separator;
    faceLandmarkPath += "facial-landmarks-35-adas-0002";
    faceLandmarkPath += std::filesystem::path::preferred_separator;

    // Emotion Model Path
    std::string emotionsPath = modelPath;
    emotionsPath += std::filesystem::path::preferred_separator;
    emotionsPath += "emotions-recognition-retail-0003";
    emotionsPath += std::filesystem::path::preferred_separator;

    faceDetectionPath += preStr.at(face_precision_);
    headPosePath += preStr.at(head_precision_);
    faceLandmarkPath += preStr.at(landmark_precision_);
    emotionsPath += preStr.at(emotion_precision_);

    faceDetectionPath += std::filesystem::path::preferred_separator;
    faceDetectionPath += "face-detection-retail-0005.xml";
    headPosePath += std::filesystem::path::preferred_separator;
    headPosePath += "head-pose-estimation-adas-0001.xml";
    faceLandmarkPath += std::filesystem::path::preferred_separator;
    faceLandmarkPath += "facial-landmarks-35-adas-0002.xml";
    emotionsPath += std::filesystem::path::preferred_separator;
    emotionsPath+= "emotions-recognition-retail-0003.xml";

    // Check that model files exist
    if (!std::filesystem::exists(faceDetectionPath)) {
        std::cerr << "Face Detection Model File is Missing - " << faceDetectionPath << std::endl;
        return false;
    }
    if (!std::filesystem::exists(headPosePath)) {
        std::cerr << "Head Pose Model File is Missing - " << headPosePath << std::endl;
        return false;
    }
    if (!std::filesystem::exists(faceLandmarkPath)) {
        std::cerr << "Facial Landmark Model File is Missing - " << faceLandmarkPath << std::endl;
        return false;
    }
    if (!std::filesystem::exists(emotionsPath)) {
        std::cerr << "Emotions Model File is Missing - " << emotionsPath << std::endl;
        return false;
    }

    face_detector_ = std::make_unique<FaceDetection>(faceDetectionPath, devices_.at(face_detect_device_), face_max_batch_, false, true, face_threshold_, false, 1.2f, 1.0f, 1.0f);
    head_pose_detector_ = std::make_unique<HeadPoseDetection>(headPosePath, devices_.at(head_pose_device_), head_max_batch_, false, true, false);
    facial_landmarks_detector_ = std::make_unique<FacialLandmarksDetection>(faceLandmarkPath, devices_.at(landmark_device_), landmark_max_batch_, false, true, false);
    emotion_detector_ = std::make_unique<EmotionsDetection>(emotionsPath, devices_.at(emotion_device_), emotion_max_batch_, false, true, false);

    std::set<std::string> loadedDevices;
    std::pair<std::string, std::string> cmdOptions[] = {
        {devices_.at(face_detect_device_), faceDetectionPath},          // Face Detection Model Option
        {devices_.at(head_pose_device_), headPosePath},       // Head Pose Model Options
        {devices_.at(landmark_device_), faceLandmarkPath},       // Face Landmark Model Options
        {devices_.at(emotion_device_), emotionsPath}       // Emotion Model Options
    };

    for (auto && option : cmdOptions) {
        auto deviceName = option.first;
        auto networkName = option.second;
        if (deviceName.empty() || networkName.empty()) {
            continue;
        }
        if (loadedDevices.find(deviceName) != loadedDevices.end()) {
            continue;
        }

        loadedDevices.insert(deviceName);
    }

    Load(*face_detector_).into(core_, devices_.at(face_detect_device_), false);
    Load(*head_pose_detector_).into(core_, devices_.at(head_pose_device_), false);
    Load(*facial_landmarks_detector_).into(core_, devices_.at(landmark_device_), false);
    Load(*emotion_detector_).into(core_, devices_.at(emotion_device_), false);

    head_pose_detector_->_enabled = head_pose_enabled_;
    facial_landmarks_detector_->_enabled = face_landmarks_enabled_;
    emotion_detector_->_enabled = emotion_enabled_;

    return true;
}

void HeadFaceEmotion::Process_( SignalBus const& inputs, SignalBus& outputs )
{
    // OpenVino processing is already multithreaded so no need to multithread this
    if (io_mutex_.try_lock()) { // Try lock so other threads will skip if locked instead of waiting
        cv::Mat frame;
        // Handle Input Code Here
        auto in1 = inputs.GetValue<cv::Mat>(0);
        if (!in1) {
            io_mutex_.unlock();
            return;
        }

        if (!in1->empty()) {
            in1->copyTo(frame);

            if (!last_frame_.empty()) {
                if (frame.size != last_frame_.size) {
                    init_once_ = true;
                    firstProc = true;
                }
            }

            frame.copyTo(last_frame_);

            if (init_once_) {
                init_success_ = false;
                init_success_ = InitModels();
                init_once_ = false;
            }

            if (init_success_) {
                if (firstProc) {
                    visualizer_ = std::make_unique<Visualizer>(frame.size());
                    face_detector_->enqueue(last_frame_);
                    face_detector_->submitRequest();
                    firstProc = false;
                    outputs.SetValue(0, frame);
                    io_mutex_.unlock();
                    return;
                }
                face_detector_->wait();
                face_detector_->fetchResults();
                auto prev_detection_results = face_detector_->results;
                face_detector_->enqueue(last_frame_);
                face_detector_->submitRequest();

                nlohmann::json json_out;
                nlohmann::json jHeads;
                nlohmann::json ref;
                json_out["data_type"] = "heads";
                ref["w"] = frame.cols;
                ref["h"] = frame.rows;
                json_out["ref_frame"] = ref;
                for (auto &&face: prev_detection_results) {
                    cv::Rect clippedRect = face.location & cv::Rect({0, 0}, last_frame_.size());
                    if (face.location.x > 0 && face.location.y > 0) {
                        if (face.location.width == face.location.height) {
                            cv::Mat faceProc = last_frame_(clippedRect);
                            head_pose_detector_->enqueue(faceProc);
                            facial_landmarks_detector_->enqueue(faceProc);
                            emotion_detector_->enqueue(faceProc);
                        }
                    }
                }

                // Submit
                head_pose_detector_->submitRequest();
                facial_landmarks_detector_->submitRequest();
                emotion_detector_->submitRequest();

                // Wait
                head_pose_detector_->wait();
                facial_landmarks_detector_->wait();
                emotion_detector_->wait();

                std::list<Face::Ptr> prev_faces;
                prev_faces.insert(prev_faces.begin(), faces_.begin(), faces_.end());

                faces_.clear();
                for (size_t i = 0; i < prev_detection_results.size(); i++) {
                    nlohmann::json jFace;
                    auto &result = prev_detection_results[i];
                    cv::Rect rect = result.location & cv::Rect({0, 0}, last_frame_.size());
                    Face::Ptr face;

                    nlohmann::json faceRect;
                    faceRect["x"] = result.location.x;
                    faceRect["y"] = result.location.y;
                    faceRect["w"] = result.location.width;
                    faceRect["h"] = result.location.height;
                    jFace["bbox"] = faceRect;
                    jFace["conf"] = result.confidence;

                    face = matchFace(rect, prev_faces);
                    float intensity_mean = calcMean(last_frame_(rect));
                    if ((face == nullptr) ||
                        ((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f)) {
                        face = std::make_shared<Face>(faceId++, rect);
                    } else {
                        prev_faces.remove(face);
                    }
                    jFace["id"] = face->getId();
                    face->_intensity_mean = intensity_mean;
                    face->_location = rect;

                    face->headPoseEnable((head_pose_detector_->enabled() && i < head_pose_detector_->maxBatch));
                    if (face->isHeadPoseEnabled()) {
                        face->updateHeadPose(head_pose_detector_->operator[](i));
                        HeadPoseDetection::Results hp = head_pose_detector_->operator[](i);
                        nlohmann::json headPose;
                        headPose["yaw"] = hp.angle_y;
                        headPose["pitch"] = hp.angle_p;
                        headPose["roll"] = hp.angle_r;
                        jFace["head_pose"] = headPose;
                    }
                    face->landmarksEnable((facial_landmarks_detector_->enabled() && i < facial_landmarks_detector_->maxBatch));
                    if (face->isLandmarksEnabled()) {
                        face->updateLandmarks(facial_landmarks_detector_->operator[](i));
                        nlohmann::json faceLandmarks;
                        auto normed_landmarks = facial_landmarks_detector_->operator[](i);
                        auto n_lm = normed_landmarks.size();
                        faceLandmarks["count"] = (int) (n_lm / 2);
                        nlohmann::json landmarkPoints;
                        for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
                            float normed_x = normed_landmarks[2 * i_lm];
                            float normed_y = normed_landmarks[2 * i_lm + 1];
                            int x_lm = rect.x + rect.width * normed_x;
                            int y_lm = rect.y + rect.height * normed_y;
                            landmarkPoints.emplace_back(int(x_lm));
                            landmarkPoints.emplace_back(int(y_lm));
                        }
                        faceLandmarks["coords"] = landmarkPoints;
                        jFace["face_landmarks"] = faceLandmarks;
                    }
                    face->emotionsEnable((emotion_detector_->enabled() && i < emotion_detector_->maxBatch));
                    if (face->isEmotionsEnabled()) {
                        face->updateEmotions(emotion_detector_->operator[](i));
                        auto emotion = emotion_detector_->operator[](i);
                        nlohmann::json emotions;
                        for (auto &e: emotion) {
                            emotions[e.first] = e.second;
                        }
                        jFace["emotions"] = emotions;
                    }
                    jHeads.emplace_back(jFace);
                    faces_.emplace_back(face);
                }
                json_out["data"] = jHeads;

                visualizer_->draw(frame, faces_);
                outputs.SetValue(0, frame);
                outputs.SetValue(1, json_out);
            }
        }
        io_mutex_.unlock();
    }
}

bool HeadFaceEmotion::HasGui(int interface)
{
    // This is where you tell the system if your node has any of the following interfaces: Main, Control or Other
    if (interface == (int)FlowCV::GuiInterfaceType_Controls) {
        return true;
    }

    return false;
}

void HeadFaceEmotion::UpdateGui(void *context, int interface)
{
    auto *imCurContext = (ImGuiContext *)context;
    ImGui::SetCurrentContext(imCurContext);

    // When Creating Strings for Controls use: CreateControlString("Text Here", GetInstanceCount()).c_str()
    // This will ensure a unique control name for ImGui with multiple instance of the Plugin
    if (interface == (int)FlowCV::GuiInterfaceType_Controls) {
        ImGui::TextUnformatted("Face Detection");
        ImGui::SetNextItemWidth(120);
        if (ImGui::Combo(CreateControlString("Run Face Device", GetInstanceName()).c_str(), &face_detect_device_, [](void* data, int idx, const char** out_text) {
            *out_text = ((const std::vector<std::string>*)data)->at(idx).c_str();
            return true;
        }, (void*)&devices_, (int)devices_.size())) {
            init_once_ = true;
            firstProc = true;
        }
        if (ImGui::Combo(CreateControlString("Face Precision", GetInstanceName()).c_str(), &face_precision_, "INT8\0FP16\0FP32\0\0")) {
            init_once_ = true;
            firstProc = true;
        }
        ImGui::SetNextItemWidth(120);
        if (ImGui::DragInt(CreateControlString("Face Max Batch", GetInstanceName()).c_str(), &face_max_batch_, 1.0f, 1, 16)) {
            if (face_max_batch_ < 1)
                face_max_batch_ = 1;
            init_once_ = true;
            firstProc = true;
        }
        ImGui::SetNextItemWidth(120);
        if (ImGui::DragFloat(CreateControlString("Face Threshold", GetInstanceName()).c_str(), &face_threshold_, 0.05f, 0.0f, 1.0f)) {
            init_once_ = true;
            firstProc = true;
        }
        ImGui::Separator();

        ImGui::TextUnformatted("Head Pose");
        if (ImGui::Checkbox(CreateControlString("Head Pose Enabled", GetInstanceName()).c_str(), &head_pose_enabled_)) {
            if (init_success_)
                head_pose_detector_->_enabled = head_pose_enabled_;
        }
        if (head_pose_enabled_) {
            ImGui::SetNextItemWidth(120);
            if (ImGui::Combo(CreateControlString("Run Head Pose Device", GetInstanceName()).c_str(), &head_pose_device_, [](void *data, int idx, const char **out_text) {
                *out_text = ((const std::vector<std::string> *) data)->at(idx).c_str();
                return true;
            }, (void *) &devices_, (int) devices_.size())) {
                init_once_ = true;
                firstProc = true;
            }
            if (ImGui::Combo(CreateControlString("Head Precision", GetInstanceName()).c_str(), &head_precision_, "INT8\0FP16\0FP32\0\0")) {
                init_once_ = true;
                firstProc = true;
            }
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragInt(CreateControlString("Head Max Batch", GetInstanceName()).c_str(), &head_max_batch_, 1.0f, 1, 16)) {
                if (head_max_batch_ < 1)
                    head_max_batch_ = 1;
                init_once_ = true;
                firstProc = true;
            }
            ImGui::Separator();
        }

        ImGui::TextUnformatted("Face Landmarks");
        if (ImGui::Checkbox(CreateControlString("Landmarks Enabled", GetInstanceName()).c_str(), &face_landmarks_enabled_)) {
            if (!face_landmarks_enabled_)
                emotion_enabled_ = false;
            if (init_success_) {
                emotion_detector_->_enabled = emotion_enabled_;
                facial_landmarks_detector_->_enabled = face_landmarks_enabled_;
            }

        }
        if (face_landmarks_enabled_) {
            ImGui::SetNextItemWidth(120);
            if (ImGui::Combo(CreateControlString("Run Landmarks Device", GetInstanceName()).c_str(), &landmark_device_, [](void *data, int idx, const char **out_text) {
                *out_text = ((const std::vector<std::string> *) data)->at(idx).c_str();
                return true;
            }, (void *) &devices_, (int) devices_.size())) {
                init_once_ = true;
                firstProc = true;
            }
            if (ImGui::Combo(CreateControlString("Landmark Precision", GetInstanceName()).c_str(), &landmark_precision_, "INT8\0FP16\0FP32\0\0")) {
                init_once_ = true;
                firstProc = true;
            }
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragInt(CreateControlString("Landmark Max Batch", GetInstanceName()).c_str(), &landmark_max_batch_, 1.0f, 1, 16)) {
                if (landmark_max_batch_ < 1)
                    landmark_max_batch_ = 1;
                init_once_ = true;
                firstProc = true;
            }
            ImGui::Separator();
            ImGui::TextUnformatted("Emotion Detection");
            if (ImGui::Checkbox(CreateControlString("Emotions Enabled", GetInstanceName()).c_str(), &emotion_enabled_)) {
                if (init_success_)
                    emotion_detector_->_enabled = emotion_enabled_;
            }
            if (emotion_enabled_) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::Combo(CreateControlString("Run Emotions Device", GetInstanceName()).c_str(), &emotion_device_, [](void *data, int idx, const char **out_text) {
                    *out_text = ((const std::vector<std::string> *) data)->at(idx).c_str();
                    return true;
                }, (void *) &devices_, (int) devices_.size())) {
                    init_once_ = true;
                    firstProc = true;
                }
                if (ImGui::Combo(CreateControlString("Emotion Precision", GetInstanceName()).c_str(), &emotion_precision_, "INT8\0FP16\0FP32\0\0")) {
                    init_once_ = true;
                    firstProc = true;
                }
                ImGui::SetNextItemWidth(120);
                if (ImGui::DragInt(CreateControlString("Emotion Max Batch", GetInstanceName()).c_str(), &emotion_max_batch_, 1.0f, 1, 16)) {
                    if (emotion_max_batch_ < 1)
                        emotion_max_batch_ = 1;
                    init_once_ = true;
                    firstProc = true;
                }
            }
        }


    }

}

int HeadFaceEmotion::MatchDeviceIndex(std::string &devStr, int devIdx)
{
    if (devIdx > devices_.size() - 1)
        devIdx = 0;
    if (devStr != devices_.at(devIdx)) {
        bool foundMatch = false;
        for (int i = 0; i < devices_.size(); i++) {
            if (devices_.at(i) == devStr) {
                return i;
            }
        }
    }
    return 0;
}

std::string HeadFaceEmotion::GetState()
{
    using namespace nlohmann;

    json state;

    state["face_device_idx"] = face_detect_device_;
    state["face_device"] = devices_.at(face_detect_device_);
    state["face_max_batch"] = face_max_batch_;
    state["face_precision"] = face_precision_;
    state["face_threshold"] = face_threshold_;
    state["head_enabled"] = head_pose_enabled_;
    state["head_device_idx"] = head_pose_device_;
    state["head_device"] = devices_.at(head_pose_device_);
    state["head_max_batch"] = head_max_batch_;
    state["head_precision"] = head_precision_;
    state["landmark_enabled"] = face_landmarks_enabled_;
    state["landmark_device_idx"] = landmark_device_;
    state["landmark_device"] = devices_.at(landmark_device_);
    state["landmark_max_batch"] = landmark_max_batch_;
    state["landmark_precision"] = landmark_precision_;
    state["emotion_enabled"] = emotion_enabled_;
    state["emotion_device_idx"] = emotion_device_;
    state["emotion_device"] = devices_.at(emotion_device_);
    state["emotion_max_batch"] = emotion_max_batch_;
    state["emotion_precision"] = emotion_precision_;

    std::string stateSerialized = state.dump(4);

    return stateSerialized;
}

void HeadFaceEmotion::SetState(std::string &&json_serialized)
{
    using namespace nlohmann;

    json state = json::parse(json_serialized);

    if (state.contains("face_device_idx"))
        face_detect_device_ = state["face_device_idx"].get<int>();
    if (state.contains("face_device")) {
        auto devStr = state["face_device"].get<std::string>();
        face_detect_device_ = MatchDeviceIndex(devStr, face_detect_device_);
    }
    if (state.contains("face_max_batch"))
        face_max_batch_ = state["face_max_batch"].get<int>();
    if (state.contains("face_precision"))
        face_precision_ = state["face_precision"].get<int>();
    if (state.contains("face_threshold"))
        face_threshold_ = state["face_threshold"].get<float>();

    if (state.contains("head_device_idx"))
        head_pose_device_ = state["head_device_idx"].get<int>();
    if (state.contains("head_device")) {
        auto devStr = state["head_device"].get<std::string>();
        head_pose_device_ = MatchDeviceIndex(devStr, head_pose_device_);
    }
    if (state.contains("head_enabled"))
        head_pose_enabled_ = state["head_enabled"].get<int>();
    if (state.contains("head_max_batch"))
        head_max_batch_ = state["head_max_batch"].get<int>();
    if (state.contains("head_precision"))
        head_precision_ = state["head_precision"].get<int>();

    if (state.contains("landmark_device_idx"))
        landmark_device_ = state["landmark_device_idx"].get<int>();
    if (state.contains("landmark_device")) {
        auto devStr = state["landmark_device"].get<std::string>();
        landmark_device_ = MatchDeviceIndex(devStr, landmark_device_);
    }
    if (state.contains("landmark_enabled"))
        face_landmarks_enabled_ = state["landmark_enabled"].get<int>();
    if (state.contains("landmark_max_batch"))
        landmark_max_batch_ = state["landmark_max_batch"].get<int>();
    if (state.contains("landmark_precision"))
        landmark_precision_ = state["landmark_precision"].get<int>();

    if (state.contains("emotion_device_idx"))
        emotion_device_ = state["emotion_device_idx"].get<int>();
    if (state.contains("emotion_device")) {
        auto devStr = state["emotion_device"].get<std::string>();
        emotion_device_ = MatchDeviceIndex(devStr, emotion_device_);
    }
    if (state.contains("emotion_enabled"))
        emotion_enabled_ = state["emotion_enabled"].get<int>();
    if (state.contains("emotion_max_batch"))
        emotion_max_batch_ = state["emotion_max_batch"].get<int>();
    if (state.contains("emotion_precision"))
        emotion_precision_ = state["emotion_precision"].get<int>();
}
