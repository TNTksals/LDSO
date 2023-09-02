# Point hessian is the internal structure of a map point

float32 u
float32 v
float32 idepth_scaled 
float32 idepth_hessian
float32 max_rel_baseline

# the last two residuals
PointFrameResidualData[2] point_frame_residual
int32[2] res_state

float32[8] color   # colors in host frame