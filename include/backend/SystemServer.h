#pragma once
#ifndef LDSO_SYSTEM_SERVER_H_
#define LDSO_SYSTEM_SERVER_H_

#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <iomanip>

// #include <boost/thread.hpp>
// #include <queue>
// #include <condition_variable>
#include <glog/logging.h>

#include "Frame.h"
#include "Point.h"
#include "Feature.h"
#include "Camera.h"
#include "Map.h"

#include "frontend/DSOViewer.h"
#include "frontend/LoopClosing.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

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

using namespace std;
using namespace ldso;
using namespace ldso::internal;

namespace ldso
{

    class SystemServer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        SystemServer(shared_ptr<ORBVocabulary> voc, ros::NodeHandle &nh);


        // ~SystemServer()
        // {
        //     // 等待回环检测线程和地图匹配线程结束
        //     m_loopClosingThread.join();
        //     m_mapMatchingThread.join();
        // }

        shared_ptr<CoarseDistanceMap> GetDistanceMap()
        {
            return coarseDistanceMap;
        }

        vector<shared_ptr<Frame>> GetActiveFrames()
        {
            unique_lock<mutex> lck(framesMutex);
            return frames;
        }

        void RefreshGUI()
        {
            if (viewer)
                viewer->refreshAll();
        }

    private:
        // 关键帧信息回调函数
        void keyframeCallback(const ldso::KeyFrame &kf_msg);

        void deliverKeyFrame(shared_ptr<Frame> frame);

        // // 回环检测线程函数
        // void loopClosingThreadFunc()
        // {
        //     while (true)
        //     {
        //         // 从回环检测队列中取出帧
        //         shared_ptr<Frame> frame = nullptr;
        //         {
        //             unique_lock<mutex> lock(m_loopClosingMutex);
        //             while (m_loopClosingQueue.empty())
        //                 m_loopClosingCond.wait(lock);
        //             frame = m_loopClosingQueue.front();
        //             m_loopClosingQueue.pop();
        //         }

        //         // 使用BoW模型从地图中查找匹配候选帧
        //         vector<shared_ptr<Frame>> candidates = m_map->getLoopCandidates(frame, m_voc);

        //         // 在候选帧中寻找回环
        //         if (m_loopClosing != nullptr)
        //             m_loopClosing->detectLoop(frame, candidates);
        //     }
        // }

        // // 地图匹配线程函数
        // void mapMatchingThreadFunc()
        // {
        //     while (true)
        //     {
        //         // 从地图匹配队列中取出帧
        //         shared_ptr<Frame> frame = nullptr;
        //         {
        //             unique_lock<mutex> lock(m_mapMatchingMutex);
        //             while (m_mapMatchingQueue.empty())
        //                 m_mapMatchingCond.wait(lock);
        //             frame = m_mapMatchingQueue.front();
        //             m_mapMatchingQueue.pop();
        //         }

        //         // 使用BoW模型从地图中查找匹配候选帧
        //         vector<shared_ptr<Frame>> candidates = m_map->getMapMatchingCandidates(frame, m_voc);

        //         // 在候选帧中寻找匹配
        //         for (auto &candidate : candidates)
        //         {
        //             // 计算当前帧和候选帧之间的Sim(3)变换
        //             Sophus::Sim3d T;
        //             if (computeSim3(frame, candidate, T))
        //             {
        //                 // 将匹配结果添加到地图中
        //                 m_map->addMatch(frame, candidate, T);
        //             }
        //         }
        //     }
        // }

        // // 计算当前帧和候选帧之间的Sim(3)变换
        // bool computeSim3(shared_ptr<Frame> frame, shared_ptr<Frame> candidate, Sophus::Sim3d &T)
        // {
        //     // 提取ORB特征点和描述子
        //     vector<KeyPoint> kps1, kps2;
        //     Mat descs1, descs2;
        //     kps1 = frame->mvKeys;
        //     kps2 = candidate->mvKeys;
        //     descs1 = frame->mDescriptors;
        //     descs2 = candidate->mDescriptors;

        //     // 使用BoW模型进行特征点匹配
        //     vector<DMatch> matches;
        //     m_bowMatcher.match(descs1, descs2, matches);

        //     // 选取匹配点对
        //     vector<Point2f> pts1, pts2;
        //     for (auto &match : matches)
        //     {
        //         pts1.push_back(kps1[match.queryIdx].pt);
        //         pts2.push_back(kps2[match.trainIdx].pt);
        //     }

        //     // 计算Sim(3)变换
        //     if (pts1.size() >= 6)
        //     {
        //         Mat inliers;
        //         T = Sophus::Sim3d::estimate(pts1, pts2, inliers, 3, 0.99);
        //         return true;
        //     }
        //     else
        //     {
        //         return false;
        //     }
        // }

    private:
        ros::NodeHandle nh;
        ros::Subscriber kf_sub;

    public:
        shared_ptr<Camera> Hcalib = nullptr; // calib information


        shared_ptr<ORBVocabulary> voc;
        // ORB m_orb;
        // shared_ptr<Map> m_map;
        // shared_ptr<LoopClosing> m_loopClosing;
        // queue<shared_ptr<Frame>> m_loopClosingQueue;
        // queue<shared_ptr<Frame>> m_mapMatchingQueue;
        // mutex m_loopClosingMutex;
        // mutex m_mapMatchingMutex;
        // condition_variable m_loopClosingCond;
        // condition_variable m_mapMatchingCond;
        // boost::thread m_loopClosingThread;
        // boost::thread m_mapMatchingThread;
        // Ptr<DescriptorMatcher> m_bowMatcher = DescriptorMatcher::create("BruteForce-Hamming");
    private:
        shared_ptr<CoarseDistanceMap> coarseDistanceMap = nullptr; // coarse distance map

        // all frames
        std::vector<shared_ptr<Frame>> frames; // all active frames, ONLY changed in marginalizeFrame and addFrame.
        mutex framesMutex;                     // mutex to lock frame read and write because other places will use this information

    public:
        shared_ptr<Map> globalMap = nullptr; // global map

    public:
        // ========================== loop closing ==================================== //
        shared_ptr<ORBVocabulary> vocab = nullptr;      // vocabulary
        shared_ptr<LoopClosing> loop_closing = nullptr; // loop closing

        // ========================= visualization =================================== //
    public:
        void setViewer(shared_ptr<PangolinDSOViewer> v)
        {
            viewer = v;
            if (viewer)
                viewer->setMap(globalMap);
        }

    private:
        shared_ptr<PangolinDSOViewer> viewer = nullptr;

    };

} // namespace ldso

#endif