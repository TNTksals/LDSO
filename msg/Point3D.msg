# Point message
uint64 id   # id
uint32 status   # status of this point
geometry_msgs/Vector3 mWorldPos   # pos in world
float32 host_feat_u
float32 host_feat_v
std_msgs/Float64MultiArray hostfeat_hostframe_tcw_opti
PointHessianData point_hessian  # hessian data