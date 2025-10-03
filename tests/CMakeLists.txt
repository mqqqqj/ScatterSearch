set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

add_executable(search search.cpp)
target_link_libraries(search ${PROJECT_NAME} -fopenmp)

add_executable(parallel_search parallel_search.cpp)
target_link_libraries(parallel_search ${PROJECT_NAME} -fopenmp)

add_executable(multi_turn_search multi_turn_search.cpp)
target_link_libraries(multi_turn_search ${PROJECT_NAME} -fopenmp)

add_executable(pthread_search pthread_search.cpp)
target_link_libraries(pthread_search ${PROJECT_NAME} -pthread)

add_executable(pthread_sync_search pthread_sync_search.cpp)
target_link_libraries(pthread_sync_search ${PROJECT_NAME} -pthread)

add_executable(pipeline_search pipeline_search.cpp)
target_link_libraries(pipeline_search ${PROJECT_NAME} -fopenmp)

add_executable(pipeline_et_search pipeline_et_search.cpp)
target_link_libraries(pipeline_et_search ${PROJECT_NAME} -fopenmp)

add_executable(online_pipeline_et_search online_pipeline_et_search.cpp)
target_link_libraries(online_pipeline_et_search ${PROJECT_NAME} -fopenmp)

add_executable(online_parallel_search online_parallel_search.cpp)
target_link_libraries(online_parallel_search ${PROJECT_NAME} -fopenmp)

add_executable(single_query single_query.cpp)
target_link_libraries(single_query ${PROJECT_NAME} -fopenmp)

add_executable(hop_find_first_knn hop_find_first_knn.cpp)
target_link_libraries(hop_find_first_knn ${PROJECT_NAME} -fopenmp)

add_executable(tree_termination tree_termination.cpp)
target_link_libraries(tree_termination ${PROJECT_NAME} -pthread)