# cmake requirements
cmake_minimum_required(VERSION 3.28)

# Build options have to be before PROJECT(...)
SET(CMAKE_CONFIGURATION_TYPES "Release,Debug" CACHE PATH "Configuration types")
# SET(CMAKE_BUILD_TYPE Release)

# Example Project configuration
PROJECT(CUDA_EXAMPLES)

SET(CXX_STANDARD "-std=c++23")
SET(CMAKE_CXX_FLAGS "${CXX_STANDARD} -Wall -Wextra -Wshadow -Wconversion -pthread -fopenmp")

# Folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# Settings
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include")
option(LOG_DEBUG "Turn on debugging log" OFF)
mark_as_advanced(LOG_DEBUG)
if(LOG_DEBUG)
    add_definitions(-DLOG_DEBUG)
endif(LOG_DEBUG)

# core libraries
option(BUILD_WITH_CUDA "Build with NVidia CUDA" ON)
mark_as_advanced(BUILD_WITH_CUDA)
if(BUILD_WITH_CUDA)
    add_definitions(-DBUILD_WITH_CUDA)
endif(BUILD_WITH_CUDA)

# demos
option(BUILD_DEMOS "Build CUDA demo" ON)
mark_as_advanced(BUILD_DEMOS)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#additional dependencies
if(BUILD_WITH_CUDA)
    # Figure out CUDA version
    if(CMAKE_CUDA_COMPILER_LOADED)
            if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
                    set(CUDA_VERSION "${CMAKE_MATCH_1}")
            endif()
    endif()
    # CUDA flags
#    set(CMAKE_CUDA_ARCHITECTURES 86)
#    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    project(WALKERS LANGUAGES CXX CUDA)
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)
    # Check CUDA version
    if (CUDAToolkit_VERSION VERSION_LESS "12.0")
        message(FATAL_ERROR "This project requires CUDA 12.0 or higher")
    endif ()
endif(BUILD_WITH_CUDA)


# Executable output directory
SET(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Executable output directory")
mark_as_advanced(RUNTIME_OUTPUT_DIRECTORY)

# Dynamic library output directory
SET(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Dynamic library output directory")
mark_as_advanced(LIBRARY_OUTPUT_DIRECTORY)

# Static library output directory
SET(ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Static library output directory")
mark_as_advanced(ARCHIVE_OUTPUT_DIRECTORY)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_DIRECTORY})
foreach(CONFIGURATION_TYPE ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIGURATION_TYPE} CONFIGURATION_TYPE)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${RUNTIME_OUTPUT_DIRECTORY})
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${LIBRARY_OUTPUT_DIRECTORY})
        SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${ARCHIVE_OUTPUT_DIRECTORY})
endforeach(CONFIGURATION_TYPE CMAKE_CONFIGURATION_TYPES)

if(MSVC)
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR CMake)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

set(CONF_INCLUDE_DIRS "${RELATIVE_PATH}" "${CMAKE_SOURCE_DIR}")

set(CONF_LIB_DIRS "${RELATIVE_PATH}" "/build/lib")


###############################################################################
#
# CUDA demos
#
###############################################################################

if(BUILD_DEMOS)
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath ./")
        SET(DEMO_SOURCES ./main.cpp)
        ADD_EXECUTABLE(main_menu ${DEMO_SOURCES})

        file(GLOB DEMO1_CUDA_SOURCES
            "${CMAKE_SOURCE_DIR}/demos/gpu_info.cu"
        )
        ADD_EXECUTABLE(demo_gpu_info ${DEMO1_CUDA_SOURCES})
        TARGET_LINK_LIBRARIES(demo_gpu_info)

        file(GLOB CUDA_ADD_SOURCES
            "${CMAKE_SOURCE_DIR}/demos/gpu_add.cu"
        )
        ADD_EXECUTABLE(demo_CUDA_add ${CUDA_ADD_SOURCES})
        TARGET_LINK_LIBRARIES(demo_CUDA_add)

        file(GLOB CUDA_ADD_KERNEL_SOURCES
            "${CMAKE_SOURCE_DIR}/demos/gpu_add_kernel.cu"
        )
        ADD_EXECUTABLE(demo_CUDA_add_kernel ${CUDA_ADD_KERNEL_SOURCES})
        TARGET_LINK_LIBRARIES(demo_CUDA_add_kernel)

        file(GLOB CUDA_INITIALIZATION_SOURCES
            "${CMAKE_SOURCE_DIR}/demos/gpu_initialization.cu"
        )
        ADD_EXECUTABLE(demo_CUDA_initialization ${CUDA_INITIALIZATION_SOURCES})
        TARGET_LINK_LIBRARIES(demo_CUDA_initialization)

endif(BUILD_DEMOS)
