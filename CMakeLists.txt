cmake_minimum_required(VERSION 3.16)
project(ByteTrack LANGUAGES C CXX)

# The Rockchip toolchain ships GCC 8.3.0. Raise to 17 if the code needs it
# (std::filesystem additionally needs stdc++fs with this GCC).
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# compile_commands.json for clangd / editors
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ninja captures compiler output through a pipe, so force colored diagnostics
if(CMAKE_GENERATOR MATCHES "Ninja")
  add_compile_options(-fdiagnostics-color=always)
endif()
add_compile_options(-Wall)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Binaries run on the board, so don't bake host build paths into RPATH
set(CMAKE_SKIP_BUILD_RPATH ON)

# ---------------------------------------------------------------------------
# RKNN runtime (librknnmrt) - RV1106/RV1103 only
# ---------------------------------------------------------------------------

option(USE_RKNN "Link the RKNN runtime (librknnmrt)" ${CMAKE_CROSSCOMPILING})
set(RKNPU2_RUNTIME "" CACHE PATH
    "Root of the downloaded rknpu2 runtime (searched recursively for armhf-uclibc/librknnmrt.so)")

if(USE_RKNN)
  if(NOT RKNPU2_RUNTIME)
    message(FATAL_ERROR "Pass -DRKNPU2_RUNTIME=/path/to/the/rknpu2-runtime/folder")
  endif()

  # Search recursively, so it does not matter how many folder levels the zip added
  file(GLOB_RECURSE _rknn_libs "${RKNPU2_RUNTIME}/librknnmrt.so")
  list(FILTER _rknn_libs INCLUDE REGEX "armhf-uclibc")
  list(LENGTH _rknn_libs _rknn_count)
  if(_rknn_count EQUAL 0)
    message(FATAL_ERROR
      "No armhf-uclibc/librknnmrt.so found under: ${RKNPU2_RUNTIME}")
  endif()
  list(GET _rknn_libs 0 RKNN_LIB)

  # Layout: .../librknn_api/armhf-uclibc/librknnmrt.so and .../librknn_api/include/rknn_api.h
  get_filename_component(_rknn_libdir "${RKNN_LIB}" DIRECTORY)
  get_filename_component(RKNN_INCLUDE_DIR "${_rknn_libdir}/../include" ABSOLUTE)
  if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
    message(FATAL_ERROR "rknn_api.h not found in: ${RKNN_INCLUDE_DIR}")
  endif()

  message(STATUS "RKNN lib:     ${RKNN_LIB}")
  message(STATUS "RKNN headers: ${RKNN_INCLUDE_DIR}")

  add_library(rknnmrt SHARED IMPORTED)
  set_target_properties(rknnmrt PROPERTIES
    IMPORTED_LOCATION             "${RKNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${RKNN_INCLUDE_DIR}")
endif()

# ---------------------------------------------------------------------------
# ByteTrack library (src/ + include/)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE BYTETRACK_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_library(bytetrack_lib STATIC ${BYTETRACK_SRC})
target_include_directories(bytetrack_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# If lib/ holds header-only dependencies (e.g. Eigen), uncomment:
# target_include_directories(bytetrack_lib PUBLIC ${CMAKE_SOURCE_DIR}/lib)

# Eigen is header-only, so a host copy (e.g. from nix) is fine
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  target_link_libraries(bytetrack_lib PUBLIC Eigen3::Eigen)
endif()

# OpenCV from the host will NOT work when cross-compiling (wrong architecture/libc).
# Pass a build made with the same toolchain, e.g. opencv-mobile:
#   make OPENCV_DIR=/path/to/opencv/lib/cmake/opencv4
if(CMAKE_CROSSCOMPILING)
  set(OPENCV_MOBILE_DIR
      "${CMAKE_SOURCE_DIR}/lib/thirdparty/opencv-mobile-5.0.0-luckfox-pico/lib/cmake/opencv5")
  if(NOT EXISTS "${OPENCV_MOBILE_DIR}/OpenCVConfig.cmake")
    message(FATAL_ERROR "opencv-mobile not found in: ${OPENCV_MOBILE_DIR}")
  endif()
  set(OpenCV_DIR "${OPENCV_MOBILE_DIR}" CACHE PATH "OpenCV for RV1106" FORCE)
  find_package(OpenCV REQUIRED CONFIG NO_DEFAULT_PATH)
else()
  find_package(OpenCV REQUIRED)
endif()
message(STATUS "OpenCV: ${OpenCV_DIR} (${OpenCV_VERSION})")

target_include_directories(bytetrack_lib PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bytetrack_lib PUBLIC ${OpenCV_LIBS})

find_package(Threads REQUIRED)
target_link_libraries(bytetrack_lib PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

if(USE_RKNN)
  target_link_libraries(bytetrack_lib PUBLIC rknnmrt)
endif()

# ---------------------------------------------------------------------------
# App: apps/main.cpp -> bytetrack_app
# ---------------------------------------------------------------------------
add_executable(bytetrack_app ${CMAKE_SOURCE_DIR}/apps/main.cpp)
target_link_libraries(bytetrack_app PRIVATE bytetrack_lib)

# Static libstdc++/libgcc: the binary no longer depends on the board's libstdc++ version
option(STATIC_LIBSTDCXX "Link libstdc++ and libgcc statically (cross builds)" ${CMAKE_CROSSCOMPILING})
if(STATIC_LIBSTDCXX)
  target_link_options(bytetrack_app PRIVATE -static-libstdc++ -static-libgcc)
endif()
