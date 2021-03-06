cmake_minimum_required(VERSION 2.8)
project(pocket-tensor-project)

# Define build options:
option(PT_BUILD_ALL "Build all pocket-tensor artefacts" OFF)
option(PT_BUILD_TESTS "Build pocket-tensor tests" OFF)
option(PT_BUILD_BENCHMARK "Build pocket-tensor benchmark" OFF)

# Define C++ version:
if(PT_BUILD_BENCHMARK OR PT_BUILD_ALL)
    # Enable C++14:
    set(CMAKE_CXX_STANDARD 14)
else()
    # Enable C++11:
    set(CMAKE_CXX_STANDARD 11)
endif()

# Detect Clang:
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_COMPILER_IS_CLANGXX 1)
endif()

# Set compiler warnings
# (https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Available.md#compilers)
# (https://stackoverflow.com/questions/3375697/useful-gcc-flags-for-c)
# (https://stackoverflow.com/questions/5088460/flags-to-enable-thorough-and-verbose-g-warnings):
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANGXX)
    add_definitions("-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wcast-align -Wcast-qual -Wunused
	-Woverloaded-virtual -Wdouble-promotion -Wformat=2 -Wfloat-equal -Wpointer-arith
	-Wwrite-strings -Wswitch-enum -Wunreachable-code -Winit-self -Wuninitialized
	-Wno-unused-result -Wctor-dtor-privacy -Wdisabled-optimization -Wmissing-declarations
        -Wredundant-decls -Wno-unused")
endif()

# Add fast math support
# (https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do):
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANGXX)
    add_definitions("-ffast-math")
elseif(MSVC)
    add_definitions("/fp:fast")
endif()

# Add native CPU features support:
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANGXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Android")
        # See documentation
    else()
        add_definitions("-march=nehalem -mavx")
        if(WANT_AVX512)
            add_definitions("-mavx2 -mfma -mavx512f -mavx512ifma")
            add_definitions("-DPT_FMADD_ENABLE=1")
            add_definitions("-DSIMDPP_ARCH_X86_AVX512F=1")
            message(STATUS "Enable AVX512+FMA")
        endif()
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
            add_definitions("-mfpu=neon")
        endif()
    endif()
elseif(MSVC)
    add_definitions("/arch:AVX")
endif()

if(CMAKE_COMPILER_IS_GNUCC)
    # smr specific compiler migration flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC")
endif()

# Define libsimdpp library path:
set(PT_LIBSIMDPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/libsimdpp" CACHE STRING "libsimdpp library path")

# Add library subdirectory:
add_subdirectory(lib)

# Add tests subdirectory:
if(PT_BUILD_TESTS OR PT_BUILD_ALL)
    add_subdirectory(tests)
endif()

# Add benchmark subdirectory:
if(PT_BUILD_BENCHMARK OR PT_BUILD_ALL)
    add_subdirectory(benchmark)
endif()
