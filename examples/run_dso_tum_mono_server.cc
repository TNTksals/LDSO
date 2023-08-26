/**
 * 这是一段接收KF消息的例程
 */

#include <ros/ros.h>

#include "ldso/CameraData.h"
#include "ldso/FrameHessianData.h"
#include "ldso/PointFrameResidualData.h"
#include "ldso/PointHessianData.h"
#include "ldso/PointData.h"
#include "ldso/FeatureData.h"
#include "ldso/FrameData.h"
#include "ldso/RelPoseData.h"
#include "ldso/KeyFrame.h"

#include <thread>
#include <clocale>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>

#include <glog/logging.h>

#include "frontend/FullSystem.h"
#include "DatasetReader.h"

#include "frontend/LoopClosing.h"
#include "internal/IndexThreadReduce.h"
#include "backend/SystemServer.h"

/*********************************************************************************
 * This program demonstrates how to run LDSO in TUM-Mono dataset
 * which is the default dataset in DSO and quite difficult because of low texture
 * Please specify the dataset directory below or by command line parameters
 *********************************************************************************/

using namespace std;
using namespace ldso;

std::string vignette = "/media/gaoxiang/Data1/Dataset/TUM-MONO/sequence_31/vignette.png";
std::string gammaCalib = "/media/gaoxiang/Data1/Dataset/TUM-MONO/sequence_31/pcalib.txt";
std::string source = "/media/gaoxiang/Data1/Dataset/TUM-MONO/sequence_31/";
std::string calib = "/media/gaoxiang/Data1/Dataset/TUM-MONO/sequence_31/camera.txt";
std::string output_file = "./results.txt";
std::string vocPath = "./vocab/orbvoc.dbow3";

double rescale = 1;
bool reversePlay = false;
bool disableROS = false;
int startIdx = 0;
int endIdx = 100000;
bool prefetch = false;
float playbackSpeed = 0; // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload = false;
bool useSampleOutput = false;

using namespace std;
using namespace ldso;

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "run_tmu_mono_server");
    ros::NodeHandle nh("~");

    shared_ptr<ORBVocabulary> voc(new ORBVocabulary());
    voc->load(vocPath);

    shared_ptr<SystemServer> system_server(new SystemServer(voc, nh));

    shared_ptr<PangolinDSOViewer> viewer = nullptr;
    if (!disableAllDisplay)
    {
        viewer = shared_ptr<PangolinDSOViewer>(new PangolinDSOViewer(wG[0], hG[0], false));
        system_server->setViewer(viewer);
    }
    else
    {
        LOG(INFO) << "visualization is disabled!" << endl;
    }

    ROS_INFO("start receving keyframe...");

    ros::spin();

    LOG(INFO) << "EXIT NOW!";
    return 0;
}
