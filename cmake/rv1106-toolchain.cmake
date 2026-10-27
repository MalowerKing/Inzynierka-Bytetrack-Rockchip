# cmake/rv1106-toolchain.cmake
# Luckfox Pico (RV1106): ARMv7 hard-float, uClibc, GCC 8.3.0.
#
# RV1106_TOOLCHAIN = directory that contains bin/arm-rockchip830-linux-uclibcgnueabihf-gcc
# Taken from -DRV1106_TOOLCHAIN=... or from the environment variable of the same name
# (the Makefile wrapper exports it).

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

if(NOT RV1106_TOOLCHAIN AND DEFINED ENV{RV1106_TOOLCHAIN})
  set(RV1106_TOOLCHAIN "$ENV{RV1106_TOOLCHAIN}")
endif()

if(NOT RV1106_TOOLCHAIN)
  message(FATAL_ERROR
    "RV1106_TOOLCHAIN is not set. Point it at the directory whose bin/ "
    "contains arm-rockchip830-linux-uclibcgnueabihf-gcc")
endif()

set(_RV1106_PREFIX "${RV1106_TOOLCHAIN}/bin/arm-rockchip830-linux-uclibcgnueabihf")

if(NOT EXISTS "${_RV1106_PREFIX}-gcc")
  message(FATAL_ERROR "Compiler not found: ${_RV1106_PREFIX}-gcc")
endif()

set(CMAKE_C_COMPILER   "${_RV1106_PREFIX}-gcc")
set(CMAKE_CXX_COMPILER "${_RV1106_PREFIX}-g++")
set(CMAKE_AR           "${_RV1106_PREFIX}-ar"     CACHE FILEPATH "")
set(CMAKE_RANLIB       "${_RV1106_PREFIX}-ranlib" CACHE FILEPATH "")
set(CMAKE_STRIP        "${_RV1106_PREFIX}-strip"  CACHE FILEPATH "")

# CMake re-reads this file inside try_compile, so hand the variable over to it
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RV1106_TOOLCHAIN)

# Host programs (cmake, python, ...) always come from the host
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
