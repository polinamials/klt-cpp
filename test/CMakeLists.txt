add_executable(klt-test 
    main.cpp
)

target_link_libraries(klt-test PRIVATE klt-cpp)

add_test(NAME klt-test COMMAND klt-test --image_path=${CMAKE_CURRENT_BINARY_DIR})

if(MSVC)
    target_compile_options(klt-test PRIVATE /W4 /WX)
else()
    target_compile_options(klt-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
