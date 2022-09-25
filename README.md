# FlowCV OpenVino Plugins

Adds OpenVino inference capabilities to [FlowCV](https://github.com/FlowCV-org/FlowCV)

Current Plugin List:
* 2D Human Pose Estimation
* Face, Head Pose, Facial Landmarks, Emotion Detection

More coming soon!

---

### Build Instructions

Prerequisites:
* [OpenVino Development Tools](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) - Currently Tested/Supported
  Version - 2021.4.2 LTS
* Clone [FlowCV](https://github.com/FlowCV-org/FlowCV) repo.

Follow the installation and build steps for OpenVino.

Make sure you build the demos (this is the easiest way to build all the necessary dependency libraries).

1. Clone this repo
2. cd to the repo directory
3. Run the following commands:

In the OpenVino install folder there is a bin folder that contains a septupvars script, you will need to run that first before building.

On Windows:
```shell
c:\path\to\openvino\bin\setvars.bat
```
On Linux:
```shell
source /path/to/openvino/bin/setupvars.sh
```

Now Do This:
```shell
mkdir Build
cd Build
cmake .. -DFlowCV_DIR=/path/to/FlowCV/FlowCV_SDK -DOpenVinoBuild_DIR=/path/to/demo/build/folder
make
```

---

### Usage

Once compiled you will also need to make sure you have setup the openvino vars before running FlowCV Editor or else you will need to copy over all of the dependency libraries into the plugin folder for it to work.

You will also need the following models (Use the OpenVino Model Downloader):
* architecture_type = openpose
  - human-pose-estimation-0001
* architecture_type = ae
  - human-pose-estimation-0002
  - human-pose-estimation-0003
  - human-pose-estimation-0004
  - human-pose-estimation-0005
  - human-pose-estimation-0006
  - human-pose-estimation-0007

Supports FP32, FP16 and INT8

