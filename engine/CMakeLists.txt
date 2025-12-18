cmake_minimum_required(VERSION 3.20)
project(engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(engine STATIC
    src/dummy.cpp
        src/inst.cpp
)

target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(engine PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_executable(demo apps/demo.cpp)
target_link_libraries(demo PRIVATE engine)
