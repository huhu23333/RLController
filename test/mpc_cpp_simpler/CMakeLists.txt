cmake_minimum_required(VERSION 3.10)
project(MPCController CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找 Ceres 库
find_package(Ceres REQUIRED)

# 创建共享库
add_library(mpc_controller SHARED
    mpc_controller.cpp
    mpc_c_api.cpp
)

target_include_directories(mpc_controller PRIVATE ${CERES_INCLUDE_DIRS})
target_link_libraries(mpc_controller ${CERES_LIBRARIES})

# 输出库到当前目录
set_target_properties(mpc_controller PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
