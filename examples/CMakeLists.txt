# TUM_MONO dataset
add_executable(client_tum_mono client_tum_mono.cc)

target_link_libraries(client_tum_mono
  ldso 
  ${THIRD_PARTY_LIBS}
)

add_dependencies(client_tum_mono ${PROJECT_NAME}_generate_messages_cpp)



add_executable(run_dso_tum_mono_server run_dso_tum_mono_server.cc)

target_link_libraries(run_dso_tum_mono_server
  ldso 
  ${THIRD_PARTY_LIBS}
)
add_dependencies(run_dso_tum_mono_server ${PROJECT_NAME}_generate_messages_cpp)



# EUROC dataset
add_executable(client_euroc client_euroc.cc)

target_link_libraries(client_euroc
  ldso 
  ${THIRD_PARTY_LIBS}
)
add_dependencies(client_euroc ${PROJECT_NAME}_generate_messages_cpp)



# Kitti dataset
add_executable(client_kitti client_kitti.cc)
target_link_libraries(client_kitti
  ldso 
  ${THIRD_PARTY_LIBS}
)
add_dependencies(client_kitti ${PROJECT_NAME}_generate_messages_cpp)


# 
