#include "backend/SystemServer.h"

using namespace ldso;
using namespace ldso::internal;

namespace ldso
{

    SystemServer::SystemServer(shared_ptr<ORBVocabulary> voc, ros::NodeHandle &nh) : coarseDistanceMap(new CoarseDistanceMap(wG[0], hG[0])),
                                                                                     Hcalib(new Camera(fxG[0], fyG[0], cxG[0], cyG[0])),
                                                                                     globalMap(new Map(this)),
                                                                                     vocab(voc)
    {
        this->nh = ros::NodeHandle(nh, "system_server");

        Hcalib->CreateCH(Hcalib);

        loop_closing = shared_ptr<LoopClosing>(new LoopClosing(this));
        if (setting_fastLoopClosing)
            LOG(INFO) << "Use fast loop closing" << endl;

        this->kf_sub = this->nh.subscribe("/keyframe", 1000, &SystemServer::keyframeCallback, this);


        // // 启动回环检测线程
        // m_loopClosingThread = boost::thread(&SystemServer::loopClosingThreadFunc, this);

        // // 启动地图匹配线程
        // m_mapMatchingThread = boost::thread(&SystemServer::mapMatchingThreadFunc, this);
        // // 启动回环检测线程
        // m_loopClosingThread = boost::thread(&SystemServer::loopClosingThreadFunc, this);
        // // 启动地图匹配线程
        // m_mapMatchingThread = boost::thread(&SystemServer::mapMatchingThreadFunc, this);
    }

    void SystemServer::keyframeCallback(const ldso::KeyFrame &kf_msg)
    {
        // ============================= KeyFrame(keyframe ID, Timestamp) ============================= //
        shared_ptr<Frame> frame(new Frame(kf_msg.frame.time_stamp));
        // shared_ptr<Frame> frame = std::make_shared<Frame>(kf_msg.frame.time_stamp);
        frame->id = kf_msg.frame.id;
        frame->kfId = kf_msg.frame.kf_id;
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        memcpy(T.data(), kf_msg.frame.tcw_opti.data.data(), sizeof(double) * 16);
        frame->setPoseOpti(Sim3(T));
        // cout << T << endl;
        // frame->frameHessian.reset(new FrameHessian(frame));
        frame->frameHessian = shared_ptr<FrameHessian>(new FrameHessian(frame));
        // frame->frameHessian = std::make_shared<FrameHessian>(frame);
        frame->frameHessian->frameID = kf_msg.frame_hessian.frame_id;
        ROS_INFO("grab keyframe id: %ld.", frame->kfId);

        // =========================== Features(Inverse depth, ORB descriptors .etc) =========================== //
        for (auto &fd : kf_msg.frame.features)
        {   
            // shared_ptr<Feature> feat = std::make_shared<Feature>();
            shared_ptr<Feature> feat(new Feature());
            feat->status = static_cast<Feature::FeatureStatus>(fd.status);
            feat->angle = fd.angle;
            feat->score = fd.score;
            feat->isCorner = fd.isCorner;
            feat->level = fd.level;
            feat->uv[0] = fd.position_x;
            feat->uv[1] = fd.position_y;
            feat->invD = fd.invD;
            memcpy(feat->descriptor, fd.descriptor.data(), sizeof(uchar) * 32);

            if (fd.point.id != -1)
            {
                feat->point = std::make_shared<Point>();
                // feat->point = std::shared_ptr<Point>(new Point());
                feat->point->id = fd.point.id;
                feat->point->status = static_cast<Point::PointStatus>(fd.point.status);
                feat->point->mWorldPos[0] = fd.point.mWorldPos.x;
                feat->point->mWorldPos[1] = fd.point.mWorldPos.y;
                feat->point->mWorldPos[2] = fd.point.mWorldPos.z;
                // ROS_INFO("grab point id: %ld.", feat->point->id);

                feat->point->mpPH = std::make_shared<PointHessian>();
                // feat->point->mpPH = std::shared_ptr<PointHessian>(new PointHessian());
                for (int i = 0; i < 2; ++i)
                {
                    // feat->point->mpPH->lastResiduals[i].first = std::shared_ptr<PointFrameResidual>(new PointFrameResidual());
                    feat->point->mpPH->lastResiduals[i].first = std::make_shared<PointFrameResidual>();
                    feat->point->mpPH->lastResiduals[i].first->target_id = fd.point.point_hessian.residuals[i].frame_hessian.frame_id;
                    feat->point->mpPH->lastResiduals[i].first->state_state = static_cast<internal::ResState>(fd.point.point_hessian.residuals[i].state_state);
                    feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[0] = fd.point.point_hessian.residuals[i].centerProjectedTo.x;
                    feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[1] = fd.point.point_hessian.residuals[i].centerProjectedTo.y;
                    feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[2] = fd.point.point_hessian.residuals[i].centerProjectedTo.z;
                    feat->point->mpPH->lastResiduals[i].second = static_cast<internal::ResState>(fd.point.point_hessian.res_state[i]);
                }
                // ROS_INFO("grab point hessian.");

                if (fd.point.host_feat_u != -1 || fd.point.host_feat_v != -1)
                {
                    feat->point->host_feature = std::make_shared<Feature>();
                    // feat->point->host_feature = std::shared_ptr<Feature>(new Feature());
                    feat->point->host_feature->uv[0] = fd.point.host_feat_u;
                    feat->point->host_feature->uv[1] = fd.point.host_feat_v;
                    feat->point->host_feature->invD = fd.point.host_feat_invD;
                    // feat->point->host_feature->host_frame = shared_ptr<Frame>(new Frame(fd.point.hostfeat_hostframe_timestamp));
                    feat->point->host_feature->host_frame = std::make_shared<Frame>(fd.point.hostfeat_hostframe_timestamp);
                    memcpy(T.data(), fd.point.hostfeat_hostframe_tcw_opti.data.data(), sizeof(double) * 16);
                    feat->point->host_feature->host_frame->setPoseOpti(Sim3(T));
                    // cout << T << endl;
                }
                // ROS_INFO("grab host feature.");
            }
            frame->features.emplace_back(feat);
        }
        ROS_INFO("grab features %ld.", frame->features.size());

        // ============================ Pose relative to covisible keyframes ============================ //
        for (auto &e : kf_msg.rel_poses)
        {
            shared_ptr<Frame> fr(new Frame(e.frame.time_stamp));
            // shared_ptr<Frame> fr = std::make_shared<Frame>(e.frame.time_stamp);
            fr->id = e.frame.id;
            fr->kfId = e.frame.kf_id;
            auto pose = Frame::RELPOSE();
            memcpy((int8_t *)&pose, e.pose.data.data(), sizeof(Frame::RELPOSE));
            // cout << pose.Tcr.matrix() << " " << pose.info << endl;
            frame->poseRel.emplace(make_pair(fr, pose));
        }
        ROS_INFO("grab relative pose: %ld.", frame->poseRel.size());

        // ============================= Camera calibration parameters ============================= //
        shared_ptr<Camera> cam(new Camera(kf_msg.cam_info.fx, kf_msg.cam_info.fy, kf_msg.cam_info.cx, kf_msg.cam_info.cy));
        // shared_ptr<Camera> cam = std::make_shared<Camera>(kf_msg.cam_info.fx, kf_msg.cam_info.fy, kf_msg.cam_info.cx, kf_msg.cam_info.cy);
        shared_ptr<CalibHessian> calib(new CalibHessian(cam));
        // shared_ptr<CalibHessian> calib = std::make_shared<CalibHessian>(cam);
        cam->mpCH = calib;
        memcpy(cam->mpCH->value_scaledf.data(), kf_msg.cam_info.value_scaledf.data.data(), sizeof(float) * 4);
        memcpy(cam->mpCH->value_scaledi.data(), kf_msg.cam_info.value_scaledi.data.data(), sizeof(float) * 4);
        // for (int i = 0; i < 4; ++i)
        //     cout << cam->mpCH->value_scaledf[i] << " ";
        // cout << endl;
        ROS_INFO("grab camera info.");

        ROS_INFO("successfully receved keyframe!");

        frames.emplace_back(frame);
        globalMap->AddKeyFrame(frame);
        loop_closing->InsertKeyFrame(frame);

        // // 将帧添加到子图中
        // m_map->addFrame(frame);

        // // 将帧添加到回环检测队列中
        // m_loopClosingQueue.emplace(frame);

        // // 将帧添加到地图匹配队列中
        // m_mapMatchingQueue.emplace(frame);
    }

}