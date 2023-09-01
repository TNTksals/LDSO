# FeaturePoint data
int32 status   # status of this feature
float32 angle   # rotation
float32 score   # shi-tomasi score
bool isCorner   # indicating if this is a corner
int32 level   # which pyramid level is the feature computed
float32 position_x   # x position
float32 position_y   # y position
PointData point   # corresponding 3D point
float32 invD   # inverse depth
uint8[32] descriptor   # ORB descriptors
bool is_ip   # the immature point
    