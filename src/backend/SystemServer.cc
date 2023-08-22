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

        // // 初始化相机参数
        // m_cam.reset(new Camera(1, 1, 0, 0));
        // m_calib.reset(new CalibHessian(m_cam));
        // m_cam->mpCH = m_calib;
        // m_calib->camera = m_cam;
       // m_cam.reset(new Camera(1, 1, 0, 0));
        // m_calib.reset(new CalibHessian(m_cam));
        // m_cam->mpCH = m_calib;
        // m_calib->camera = m_cam;

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
        shared_ptr<Frame> frame(new Frame(kf_msg.frame.time_stamp));
        frame->id = kf_msg.frame.id;
        frame->kfId = kf_msg.frame.kf_id;
        frame->frameHessian.reset(new FrameHessian(frame));
        ROS_INFO("grab keyframe id: %ld.", frame->kfId);


        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        memcpy(T.data(), kf_msg.frame.tcw_opti.data.data(), sizeof(double) * 16);
        
        frame->setPoseOpti(Sim3(T));
        cout << T << endl;

        for (auto &ft : kf_msg.frame.features)
        {
            std::shared_ptr<Feature> feat = std::make_shared<Feature>();
            assert(feat != nullptr);
            feat->status = static_cast<Feature::FeatureStatus>(ft.status);
            feat->angle = ft.angle;
            feat->score = ft.score;
            feat->isCorner = ft.isCorner;
            feat->level = ft.level;
            feat->uv[0] = ft.position_x;
            feat->uv[1] = ft.position_y;
            feat->invD = ft.invD;
            memcpy(feat->descriptor, ft.descriptor.data(), sizeof(uchar) * 32);
            if (ft.point.id != -1)
            {
                feat->point = std::make_shared<Point>();
                feat->point->id = ft.point.id;
                feat->point->status = static_cast<Point::PointStatus>(ft.point.status);
                feat->point->mWorldPos[0] = ft.point.mWorldPos.x;
                feat->point->mWorldPos[1] = ft.point.mWorldPos.y;
                feat->point->mWorldPos[2] = ft.point.mWorldPos.z;

                // if (ft.point.host_feat_u == -1 && ft.point.host_feat_v == -1)
                // {
                //     feat->point->mHostFeature.reset();
                // }
                // else
                // {
                //     feat->point->hasHostFeature = feat->point->hasHostFeatureHostFrame = true;
                //     feat->point->hostFeature_u = ft.point.host_feat_u;
                //     feat->point->hostFeature_v = ft.point.host_feat_v;
                //     Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                //     memcpy(T.data(), ft.point.hostfeat_hostframe_tcw_opti.data.data(), sizeof(double) * 16);
                //     feat->point->hostFeatureHostFrameTcw = Sim3(T);
                // }

                feat->point->mpPH = std::make_shared<PointHessian>();
                for (int i = 0; i < 2; ++i)
                {
                    feat->point->mpPH->lastResiduals[i].first = std::shared_ptr<PointFrameResidual>(new PointFrameResidual());
                    feat->point->mpPH->lastResiduals[i].first->target_id = ft.point.point_hessian.residuals[i].frame_hessian.frame_id;
                    feat->point->mpPH->lastResiduals[i].first->state_state = static_cast<ResState>(ft.point.point_hessian.residuals[i].state_state);
                    feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[0] = ft.point.point_hessian.residuals[i].centerProjectedTo.x;
                    feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[1] = ft.point.point_hessian.residuals[i].centerProjectedTo.y;
                    feat->point->mpPH->lastResiduals[i].first->centerProjectedTo[2] = ft.point.point_hessian.residuals[i].centerProjectedTo.z;
                    feat->point->mpPH->lastResiduals[i].second = static_cast<ResState>(ft.point.point_hessian.res_state[i]);
                }
            }
            else
                feat->point = nullptr;
            frame->features.emplace_back(feat);
        }
        ROS_INFO("grab features %ld.", frame->features.size());

        for (auto &e : kf_msg.rel_poses)
        {
            shared_ptr<Frame> fr(new Frame(e.frame.time_stamp));
            fr->id = e.frame.id;
            fr->kfId = e.frame.kf_id;
            auto pose = Frame::RELPOSE();
            memcpy((int8_t *)&pose, e.pose.data.data(), sizeof(pose));
            // cout << pose.Tcr.matrix() << " " << pose.info << endl;
            frame->poseRel.emplace(make_pair(fr, pose));
        }
        ROS_INFO("grab relative pose: %ld.", frame->poseRel.size());

        shared_ptr<Camera> cam(new Camera(kf_msg.cam_info.fx, kf_msg.cam_info.fy, kf_msg.cam_info.cx, kf_msg.cam_info.cy));
        shared_ptr<CalibHessian> calib(new CalibHessian(cam));
        cam->mpCH = calib;
        calib->camera = cam;
        memcpy(cam->mpCH->value_scaledf.data(), kf_msg.cam_info.value_scaledf.data.data(), sizeof(float) * 4);
        memcpy(cam->mpCH->value_scaledi.data(), kf_msg.cam_info.value_scaledi.data.data(), sizeof(float) * 4);
        // for (int i = 0; i < 4; ++i)
        //     cout << cam->mpCH->value_scaledf[i] << " ";
        // cout << endl;
        ROS_INFO("grab camera info.");

        ROS_INFO("successfully receved keyframe!");

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