cmake_minimum_required(VERSION 3.15)

project(klt-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(klt-cpp SHARED
    src/image.cpp
    src/shi_tomasi.cpp
)

target_include_directories(klt-cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()
add_subdirectory(test)

if(MSVC)
    target_compile_options(klt-cpp PRIVATE /W4 /WX)
else()
    target_compile_options(klt-cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
