# Point data
int64 id   # id
int32 status   # status of this point
geometry_msgs/Vector3 mWorldPos   # pos in world
float32 host_feat_invD
float32 host_feat_u
float32 host_feat_v
float64 hostfeat_hostframe_timestamp
std_msgs/Float64MultiArray hostfeat_hostframe_tcw_opti
PointHessianData point_hessian   # hessian data