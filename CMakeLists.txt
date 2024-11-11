cmake_minimum_required(VERSION 3.19)
project(cvnp_nano LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

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

# We need to build nanobind before we can link against it
nanobind_build_library(nanobind SHARED)
add_library(cvnp_nano src/cvnp_nano.cpp src/cvnp_nano.h src/cvnp_nano_synonyms.cpp src/cvnp_nano_synonyms.h
    src/cvnp_nano_synonyms.h)
target_include_directories(cvnp_nano PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(cvnp_nano PUBLIC opencv_core nanobind)

if (PROJECT_IS_TOP_LEVEL)
    # Add the example
    nanobind_add_module(cvnp_nano_example example/cvnp_nano_example.cpp)
    target_link_libraries(cvnp_nano_example PRIVATE opencv_core)

    # Add the tests
    enable_testing()

    find_package(Python REQUIRED COMPONENTS Development)  # Finds the Python interpreter and development files
    add_executable(test_cvnp_nano tests/test_cvnp_nano.cpp)
    target_link_libraries(test_cvnp_nano PRIVATE cvnp_nano opencv_core nanobind Python::Python)

#    target_link_libraries(test_cvnp_nano PRIVATE cvnp_nano)
#    nanobind_compile_options(test_cvnp_nano)
#    nanobind_link_options(test_cvnp_nano)
    add_test(NAME test_cvnp_nano COMMAND test_cvnp_nano)
endif()
