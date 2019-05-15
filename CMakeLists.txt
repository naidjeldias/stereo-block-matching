cmake_minimum_required(VERSION 3.8)
project(stereoMatching)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp ORBextractor.cc ORBextractor.h)
add_executable(stereoMatching ${SOURCE_FILES})

include_directories($(OpenCV_INCLUDE_DIRS))
target_link_libraries(stereoMatching ${OpenCV_LIBS})