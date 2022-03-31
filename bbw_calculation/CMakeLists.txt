cmake_minimum_required(VERSION 3.1)
project(pop_out_motion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_MOSEK       "Use MOSEK"           ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(MOSEK REQUIRED QUIET)
find_package(LIBIGL REQUIRED COMPONENTS mosek)

file(GLOB SRCFILES *.cpp)

add_executable(calc_bbw_bin calc_bbw.cpp)
target_link_libraries(calc_bbw_bin igl::core igl::opengl igl::opengl_glfw igl::mosek stdc++fs)
