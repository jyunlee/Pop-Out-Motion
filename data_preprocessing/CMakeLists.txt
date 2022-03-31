cmake_minimum_required(VERSION 3.1)
project(pop_out_motion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_MOSEK       "Use MOSEK"           ON)

find_package(LIBIGL REQUIRED QUIET)

file(GLOB SRCFILES *.cpp)

add_executable(normalize_bin normalize.cpp)
add_executable(calc_l_minv_bin calc_l_minv.cpp)

target_link_libraries(normalize_bin igl::core)
target_link_libraries(calc_l_minv_bin igl::core)

