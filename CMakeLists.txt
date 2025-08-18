cmake_minimum_required(VERSION 3.5.0)

# Check if SG200X_SDK_PATH is defined
message(STATUS "SG200X_SDK_PATH: $ENV{SG200X_SDK_PATH}")
if("$ENV{SG200X_SDK_PATH}" STREQUAL "")
  message(FATAL_ERROR "ERROR: SG200X_SDK_PATH environment variable is not defined. Please set it before running CMake.")
endif()

project(sscma-elixir C CXX)

get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)

message(STATUS "PROJECT_DIR: ${PROJECT_DIR}")

get_filename_component(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

include(${ROOT_DIR}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake)

message(STATUS "ROOT_DIR: ${ROOT_DIR}")

include(${ROOT_DIR}/cmake/project.cmake)