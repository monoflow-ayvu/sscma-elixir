file(GLOB_RECURSE srcs ${CMAKE_CURRENT_LIST_DIR}/*.c ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

set(incs ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/node
    ${CMAKE_CURRENT_SOURCE_DIR}/../components/sscma-micro/sscma-micro/sscma
    ${CMAKE_CURRENT_SOURCE_DIR}/../components/sscma-micro/porting/sophgo
    ${CMAKE_CURRENT_SOURCE_DIR}/../components/sscma-micro/porting/sophgo/sg200x
)

component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS sscma-micro cvi_rtsp
    REQUIREDS opencv_core opencv_imgcodecs opencv_imgproc
)