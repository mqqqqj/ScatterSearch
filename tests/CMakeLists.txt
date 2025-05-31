set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

add_executable(test_search search.cpp)
target_link_libraries(test_search ${PROJECT_NAME} -fopenmp)