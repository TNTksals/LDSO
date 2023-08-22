/**
 * 这是一段接收KF消息的例程
 */

#include <ros/ros.h>
#include "ldso/FrameHessianData.h"
#include "ldso/PointFrameResidualData.h"
#include "ldso/PointHessianData.h"
#include "ldso/CamInfo.h"
#include "ldso/Point3D.h"
#include "ldso/FeaturePoint.h"
#include "ldso/FrameInfo.h"
#include "ldso/RelPose.h"
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

// void callback(const ldso::KeyFrame &kf_msg)
// {
//     shared_ptr<Frame> frame(new Frame(kf_msg.frame.time_stamp));
//     frame->id = kf_msg.frame.id;
//     frame->kfId = kf_msg.frame.kf_id;
//     frame->frameHessian.reset(new FrameHessian(frame));
//     ROS_INFO("grab keyframe id: %ld.", frame->kfId);

//     for (auto &ft : kf_msg.frame.features)
//     {
//         std::shared_ptr<Feature> feat = std::make_shared<Feature>();
//         feat->status = static_cast<Feature::FeatureStatus>(ft.status);
//         feat->angle = ft.angle;
//         feat->score = ft.score;
//         feat->isCorner = ft.isCorner;
//         feat->level = ft.level;
//         feat->uv[0] = ft.position_x;
//         feat->uv[1] = ft.position_y;
//         feat->invD = ft.invD;
//         memcpy(feat->descriptor, ft.descriptor.data(), sizeof(uchar) * 32);
//         if (ft.point.id != -1)
//         {
//             feat->point = std::make_shared<Point>();
//             feat->point->id = ft.point.id;
//             feat->point->status = static_cast<Point::PointStatus>(ft.point.status);
//             feat->point->mWorldPos[0] = ft.point.mWorldPos.x;
//             feat->point->mWorldPos[1] = ft.point.mWorldPos.y;
//             feat->point->mWorldPos[2] = ft.point.mWorldPos.z;

//             feat->point->mpPH = std::make_shared<PointHessian>();
//             for (int i = 0; i < 2; ++i)
//             {
//                 feat->point->mpPH->lastResiduals[i].first = std::shared_ptr<PointFrameResidual>(new PointFrameResidual());
//                 // feat->point->mpPH->lastResiduals[i].first->target = std::make_shared<FrameHessian>();
//                 // feat->point->mpPH->lastResiduals[i].first->target.lock()->frameID = ft.point.point_hessian.residuals[i].frame_hessian.frame_id;
//                 feat->point->mpPH->lastResiduals[i].first->state_state = static_cast<ResState>(ft.point.point_hessian.residuals[i].state_state);
//                 feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[0] = ft.point.point_hessian.residuals[i].centerProjectedTo.x;
//                 feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[1] = ft.point.point_hessian.residuals[i].centerProjectedTo.y;
//                 feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[2] = ft.point.point_hessian.residuals[i].centerProjectedTo.z;
//                 feat->point->mpPH->lastResiduals[i].second = static_cast<ResState>(ft.point.point_hessian.res_state[i]);
//             }
//         }
//         else
//             feat->point = nullptr;
//         frame->features.emplace_back(feat);
//     }
//     ROS_INFO("grab features %ld.", frame->features.size());

//     for (auto &e : kf_msg.rel_poses)
//     {
//         shared_ptr<Frame> fr(new Frame(e.frame.time_stamp));
//         fr->id = e.frame.id;
//         fr->kfId = e.frame.kf_id;
//         auto pose = Frame::RELPOSE();
//         memcpy((int8_t *)&pose, e.pose.data.data(), sizeof(pose));
//         // cout << pose.Tcr.matrix() << " " << pose.info << endl;
//         frame->poseRel.emplace(make_pair(fr, pose));
//     }
//     ROS_INFO("grab relative pose: %ld.", frame->poseRel.size());

//     shared_ptr<Camera> cam(new Camera(kf_msg.cam_info.fx, kf_msg.cam_info.fy, kf_msg.cam_info.cx, kf_msg.cam_info.cy));
//     shared_ptr<CalibHessian> calib(new CalibHessian(cam));
//     cam->mpCH = calib;
//     calib->camera = cam;
//     memcpy(cam->mpCH->value_scaledf.data(), kf_msg.cam_info.value_scaledf.data.data(), sizeof(float) * 4);
//     memcpy(cam->mpCH->value_scaledi.data(), kf_msg.cam_info.value_scaledi.data.data(), sizeof(float) * 4);
//     // for (int i = 0; i < 4; ++i)
//     //     cout << cam->mpCH->value_scaledf[i] << " ";
//     // cout << endl;
//     ROS_INFO("grab camera info.");


//     ROS_INFO("successfully receved keyframe!");
// }

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
