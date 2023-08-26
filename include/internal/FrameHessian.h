#pragma once
#ifndef LDSO_FRAME_HESSIAN_H_
#define LDSO_FRAME_HESSIAN_H_

#include "Frame.h"
#include "NumTypes.h"
#include "Settings.h"
#include "AffLight.h"

#include "internal/FrameFramePrecalc.h"

using namespace std;

namespace ldso
{

    namespace internal
    {

        class PointHessian;

        class CalibHessian;

        struct FrameFramePrecalc;

        /**
         * Frame hessian is the internal structure used in dso
         * 该类是DSO（Direct Sparse Odometry）算法中的一个关键帧（keyframe）的Hessian矩阵
         * 该类包含了一些成员变量和成员函数，用于存储和处理关键帧的信息，以及计算能量（energy）和梯度（gradient）等
         */
        class FrameHessian
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            FrameHessian() = default;

            FrameHessian(shared_ptr<Frame> frame)
            {
                this->frame = frame;
            }

            ~FrameHessian()
            {
                for (int i = 0; i < pyrLevelsUsed; i++)
                {
                    delete[] dIp[i];
                    delete[] absSquaredGrad[i];
                }
            }

            // accessors
            EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPT() const
            {
                return worldToCam_evalPT;
            }

            EIGEN_STRONG_INLINE const Vec10 &get_state_zero() const
            {
                return state_zero;
            }

            EIGEN_STRONG_INLINE const Vec10 &get_state() const
            {
                return state;
            }

            EIGEN_STRONG_INLINE const Vec10 &get_state_scaled() const
            {
                return state_scaled;
            }

            // state - state0
            EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const
            {
                return get_state() - get_state_zero();
            }

            inline Vec6 w2c_leftEps() const
            {
                return get_state_scaled().head<6>();
            }

            inline AffLight aff_g2l()
            {
                return AffLight(get_state_scaled()[6], get_state_scaled()[7]);
            }

            inline AffLight aff_g2l_0() const
            {
                return AffLight(get_state_zero()[6] * SCALE_A, get_state_zero()[7] * SCALE_B);
            }

            void setStateZero(const Vec10 &state_zero);

            inline void setState(const Vec10 &state)
            {

                this->state = state;
                state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
                state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
                state_scaled[6] = SCALE_A * state[6];
                state_scaled[7] = SCALE_B * state[7];
                state_scaled[8] = SCALE_A * state[8];
                state_scaled[9] = SCALE_B * state[9];

                PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
                PRE_camToWorld = PRE_worldToCam.inverse();
            };

            inline void setStateScaled(const Vec10 &state_scaled)
            {

                this->state_scaled = state_scaled;
                state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
                state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
                state[6] = SCALE_A_INVERSE * state_scaled[6];
                state[7] = SCALE_B_INVERSE * state_scaled[7];
                state[8] = SCALE_A_INVERSE * state_scaled[8];
                state[9] = SCALE_B_INVERSE * state_scaled[9];

                PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
                PRE_camToWorld = PRE_worldToCam.inverse();
            };

            // 设置评估点的世界坐标系到相机坐标系的变换矩阵和状态变量
            inline void setEvalPT(const SE3 &worldToCam_evalPT, const Vec10 &state)
            {

                this->worldToCam_evalPT = worldToCam_evalPT;
                setState(state);
                setStateZero(state);
            };

            // set the pose Tcw 设置评估点的世界坐标系到相机坐标系的变换矩阵和仿射变换
            inline void setEvalPT_scaled(const SE3 &worldToCam_evalPT, const AffLight &aff_g2l)
            {
                Vec10 initial_state = Vec10::Zero();
                initial_state[6] = aff_g2l.a;
                initial_state[7] = aff_g2l.b;
                this->worldToCam_evalPT = worldToCam_evalPT;
                setStateScaled(initial_state);
                setStateZero(this->get_state());
            };

            /**
             * @brief create the images and gradient from original image 从原始图像创建图像和梯度
             * @param [in] HCalib camera intrinsics with hessian
             */
            void makeImages(float *image, const shared_ptr<CalibHessian> &HCalib);

            // 获取先验信息
            inline Vec10 getPrior()
            {
                Vec10 p = Vec10::Zero();
                if (frame->id == 0)
                {
                    p.head<3>() = Vec3::Constant(setting_initialTransPrior);
                    p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
                    if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
                    {
                        p.head<6>().setZero();
                    }
                    p[6] = setting_initialAffAPrior;
                    p[7] = setting_initialAffBPrior;
                }
                else
                {
                    if (setting_affineOptModeA < 0)
                    {
                        p[6] = setting_initialAffAPrior;
                    }
                    else
                    {
                        p[6] = setting_affineOptModeA;
                    }
                    if (setting_affineOptModeB < 0)
                    {
                        p[7] = setting_initialAffBPrior;
                    }
                    else
                    {
                        p[7] = setting_affineOptModeB;
                    }
                }
                p[8] = setting_initialAffAPrior;
                p[9] = setting_initialAffBPrior;
                return p;
            }

            // 获取状态变量的零先验信息
            inline Vec10 getPriorZero()
            {
                return Vec10::Zero();
            }

            // Data
            int frameID = 0;                   // key-frame ID, will be set when adding new keyframes 关键帧的ID，用于添加新的关键帧时设置
            shared_ptr<Frame> frame = nullptr; // link to original frame

            // internal structures used in DSO
            // image pyramid and gradient image
            // dIp[i] is the i-th pyramid with dIp[i][0] is the original image，[1] is dx and [2] is dy
            // by default, we have 6 pyramids, so we have dIp[0...5]
            // created in makeImages()
            Vec3f *dIp[PYR_LEVELS];  // 图像金字塔和梯度图像，其中dIp[i]是第i级金字塔，dIp[i][0]是原始图像，[1]是x方向的梯度，[2]是y方向的梯度

            // absolute squared gradient of each pyramid  每个金字塔的绝对平方梯度，仅用于像素选择（直方图等），没有NAN
            float *absSquaredGrad[PYR_LEVELS]; // only used for pixel select (histograms etc.). no NAN.

            // dI = dIp[0], the first pyramid 第一个金字塔，用于跟踪
            Vec3f *dI = nullptr; // trace, fine tracking. Used for direction select (not for gradient histograms etc.)

            // Photometric Calibration Stuff 动态设置的跟踪残差的阈值
            float frameEnergyTH = 8 * 8 * patternNum; // set dynamically depending on tracking residual
            float ab_exposure = 0;                    // the exposure time // 曝光时间

            bool flaggedForMarginalization = false; // flag for margin 标记是否需要边缘化
            Mat66 nullspaces_pose = Mat66::Zero();  // 位姿的零空间
            Mat42 nullspaces_affine = Mat42::Zero();  // 仿射变换的零空间
            Vec6 nullspaces_scale = Vec6::Zero();  // 尺度的零空间

            // variable info.  相机坐标系到世界坐标系的变换矩阵
            SE3 worldToCam_evalPT; // Tcw (in ORB-SLAM's framework)

            // state variable，[0-5] is se3, 6-7 is light param a,b 状态变量，其中state[0-5]是SE3，state[6-7]是光照参数a和b
            Vec10 state; // [0-5: worldToCam-leftEps. 6-7: a,b]

            // variables used in optimization 优化步长
            Vec10 step = Vec10::Zero();
            Vec10 step_backup = Vec10::Zero();
            Vec10 state_backup = Vec10::Zero();
            Vec10 state_zero = Vec10::Zero();
            Vec10 state_scaled = Vec10::Zero();  // 缩放后的状态变量

            // precalculated values, will be send to frame when optimization is done.
            SE3 PRE_worldToCam; // TCW 优化前的相机坐标系到世界坐标系的变换矩阵
            SE3 PRE_camToWorld; // TWC 优化前的世界坐标系到相机坐标系的变换矩阵

            std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc;

            // ======================================================================================== //
            // Energy stuffs
            // Frame status: 6 dof pose + 2 dof light param
            void takeData();                 // take data from frame hessian 从帧Hessian中获取数据
            Vec8 prior = Vec8::Zero();       // prior hessian (diagonal)
            Vec8 delta_prior = Vec8::Zero(); // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
            Vec8 delta = Vec8::Zero();       // state - state_zero.
            int idx = 0;                     // the id in the sliding window, used for constructing matricies
        };
    }
}

#endif // LDSO_FRAME_HESSIAN_H_
