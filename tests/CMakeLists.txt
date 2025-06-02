set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

add_executable(search search.cpp)
target_link_libraries(search ${PROJECT_NAME} -fopenmp)

add_executable(parallel_search parallel_search.cpp)
target_link_libraries(parallel_search ${PROJECT_NAME} -fopenmp)

add_executable(pipeline_search pipeline_search.cpp)
target_link_libraries(pipeline_search ${PROJECT_NAME} -fopenmp)

add_executable(pipeline_et_search pipeline_et_search.cpp)
target_link_libraries(pipeline_et_search ${PROJECT_NAME} -fopenmp)