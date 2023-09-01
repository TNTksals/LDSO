# keyframe info sent from the client to the server
float64 time_stamp
uint64 id   # id of this frame
uint64 kf_id   # keyframe id of this frame
std_msgs/Float64MultiArray tcw   # pose from world to camera, estimated by DSO
std_msgs/Float64MultiArray tcw_opti   # pose from world to camera optimized by global pose graph (with scale)
FeatureData[] features   # Features contained
