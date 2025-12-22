include(CMakeForceCompiler)

# The Generic system name is used for embedded targets (targets without OS) in CMake
set( CMAKE_SYSTEM_NAME          Linux )
set( CMAKE_SYSTEM_PROCESSOR     riscv )
set( ARCH riscv )

# Get the absolute path to the SDK
if(DEFINED ENV{SG200X_SDK_PATH})
    set(SG200X_SDK_PATH $ENV{SG200X_SDK_PATH})

    # Convert relative path to absolute path
    get_filename_component(SG200X_SDK_PATH_ABS "${SG200X_SDK_PATH}" ABSOLUTE)

    # Construct compiler paths
    set(COMPILER_PATH "${SG200X_SDK_PATH_ABS}/../host-tools/gcc/riscv64-linux-musl-x86_64/bin")

    # Convert compiler paths to absolute paths
    get_filename_component(CMAKE_C_COMPILER "${COMPILER_PATH}/riscv64-unknown-linux-musl-gcc" ABSOLUTE)
    get_filename_component(CMAKE_CXX_COMPILER "${COMPILER_PATH}/riscv64-unknown-linux-musl-g++" ABSOLUTE)
    get_filename_component(CMAKE_OBJCOPY "${COMPILER_PATH}/riscv64-unknown-linux-musl-objcopy" ABSOLUTE)

    message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
else()
    message(FATAL_ERROR "SG200X_SDK_PATH environment variable not set")
endif()

# To build the tests, we need to set where the target environment containing
# the required library is. On Debian-like systems, this is
# /usr/riscv64-unknown-linux-gnu-.
SET(CMAKE_FIND_ROOT_PATH ${ARM_SYSROOT_PATH})
# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# We must set the OBJCOPY setting into cache so that it's available to the
# whole project. Otherwise, this does not get set into the CACHE and therefore
# the build doesn't know what the OBJCOPY filepath is
set(CMAKE_OBJCOPY ${CMAKE_OBJCOPY} CACHE FILEPATH "The toolchain objcopy command " FORCE )

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" )
set( CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" )

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=c906fdv" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=c906fdv" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv0p7_zfh_xthead -mabi=lp64d" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv0p7_zfh_xthead -mabi=lp64d" ) 
