#add_subdirectory(Analysis)
#add_subdirectory(Conversion)
#add_subdirectory(Transform)
#add_subdirectory(llvm/Passes)

cm_find_package(CM)
include(CMDeploy)
include(CMSetupVersion)

cm_project(mlir-assigner WORKSPACE_NAME ${CMAKE_WORKSPACE_NAME} LANGUAGES ASM C CXX)

if(NOT CMAKE_CROSSCOMPILING)
    find_package(Boost COMPONENTS REQUIRED filesystem log log_setup program_options thread system)
elseif(CMAKE_CROSSCOMPILING)
    if(NOT TARGET boost)
        include(ExternalProject)
        set(Boost_LIBRARIES boost_random)
        externalproject_add(boost
                            PREFIX ${CMAKE_BINARY_DIR}/libs/boost
                            GIT_REPOSITORY git@github.com:boostorg/boost.git
                            GIT_TAG boost-1.77.0
                            BUILD_IN_SOURCE TRUE
                            CMAKE_ARGS -DCMAKE_CROSSCOMPILING_EMULATOR=${CMAKE_CROSSCOMPILING_EMULATOR} -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                            BUILD_COMMAND cmake --build . --target ${Boost_LIBRARIES}
                            INSTALL_COMMAND "")
    else()
        set(Boost_LIBRARIES boost_random)
    endif()
endif()

find_package(spdlog REQUIRED)

string(TOUPPER ${CURRENT_PROJECT_NAME} UPPER_CURRENT_PROJECT_NAME)

# get header files; only needed by CMake generators,
# expr.g., for creating proper Xcode projects
set(${CURRENT_PROJECT_NAME}_HEADERS)

# list cpp files excluding platform-dependent files
list(APPEND ${CURRENT_PROJECT_NAME}_SOURCES
     main.cpp)

add_executable(${CURRENT_PROJECT_NAME}
               ${${CURRENT_PROJECT_NAME}_HEADERS}
               ${${CURRENT_PROJECT_NAME}_SOURCES})

set_target_properties(${CURRENT_PROJECT_NAME} PROPERTIES
                      LINKER_LANGUAGE CXX
                      EXPORT_NAME ${CURRENT_PROJECT_NAME}
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED TRUE)
target_compile_options(${CURRENT_PROJECT_NAME} PRIVATE -Wno-cast-qual)

target_link_libraries(${CURRENT_PROJECT_NAME}
                      LLVMIRReader
                      LLVMCore
                      LLVMSupport
                      crypto3::algebra
                      crypto3::block
                      crypto3::blueprint
                      crypto3::codec
                      crypto3::math
                      crypto3::multiprecision
                      crypto3::pkpad
                      crypto3::pubkey
                      crypto3::random
                      crypto3::zk

                      OMCompilerUtils

                      MLIRZkMlDialect

                      marshalling::core
                      marshalling::crypto3_algebra
                      marshalling::crypto3_multiprecision
                      marshalling::crypto3_zk

                      ${Boost_LIBRARIES}
                      spdlog::spdlog_header_only)

                    
target_include_directories(${CURRENT_PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           ${CMAKE_CURRENT_SOURCE_DIR}/../include
                           ${CMAKE_SOURCE_DIR}/libs/zk-ml-dialect/include
                           ${CMAKE_BINARY_DIR}/libs/zk-ml-dialect/tablegen

                           $<$<BOOL:${Boost_FOUND}>:${Boost_INCLUDE_DIRS}>)
