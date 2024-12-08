cmake_minimum_required(VERSION 3.19)
project(cvnp_nano LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

if (NOT PROJECT_IS_TOP_LEVEL)
    message(FATAL_ERROR "
    This CMakeLists.txt is only used for the CI Tests.
    When using this in your project just include cvnp_nano/cvnp_nano.h
    ")
endif()


macro(find_nanobind)
    # From https://nanobind.readthedocs.io/en/latest/building.html

    # Shout if the user forgets to set Python_EXECUTABLE
    # (it will be set when running from pip install)
    if(NOT Python_EXECUTABLE)
        message(FATAL_ERROR "
        Please set the Python_EXECUTABLE variable to a python interpreter
        where you installed numpy and opencv. For example:

            cmake .. -DPython_EXECUTABLE=/venv/bin/python
        ")
    endif()
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
    # Detect the installed nanobind package and import it into CMake
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    message(STATUS "nanobind_ROOT: ${nanobind_ROOT}")
    find_package(nanobind CONFIG REQUIRED)
endmacro()


macro(find_opencv)
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "
        OpenCV not found: find_package(OpenCV) failed!
        While developing this extension, you may use vcpkg: see recipe vcpkg_install_opencv in justfile.

        You may use it like this:

        - If compiling as a C++ project:
            just vcpkg_install_opencv
            cmake -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ..
        - If running pip install:
            export CMAKE_PREFIX_PATH=$(pwd)/vcpkg/installed/YOUR_TRIPLET
            pip install .
    ")
    endif()
endmacro()


find_nanobind()
find_opencv()
nanobind_add_module(cvnp_nano_example example/cvnp_nano_example.cpp)
target_link_libraries(cvnp_nano_example PRIVATE opencv_core)
target_include_directories(cvnp_nano_example PRIVATE ${CMAKE_CURRENT_LIST_DIR})
# Copy the .so or .pyd module into tests/, to facilitate the tests
add_custom_command(
    TARGET cvnp_nano_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cvnp_nano_example> ${CMAKE_CURRENT_LIST_DIR}/tests
)
