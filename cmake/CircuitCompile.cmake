#---------------------------------------------------------------------------#
# Copyright (c) 2018-2022 Mikhail Komarov <nemo@nil.foundation>
# Copyright (c) 2020-2022 Nikita Kaskov <nbering@nil.foundation>
# Copyright (c) 2022 Mikhail Aksenov <maksenov@nil.foundation>
#
# Distributed under the Boost Software License, Version 1.0
# See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt
#---------------------------------------------------------------------------#

function(add_circuit_no_stdlib name)
    set(prefix ARG)
    set(noValues "")
    set(singleValues)
    set(multiValues SOURCES INCLUDE_DIRECTORIES LINK_LIBRARIES COMPILER_OPTIONS)
    cmake_parse_arguments(${prefix}
                          "${noValues}"
                          "${singleValues}"
                          "${multiValues}"
                          ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "SOURCES for ${name} circuit was not defined")
    endif()

    foreach(source ${ARG_SOURCES})
        if(NOT IS_ABSOLUTE ${include_dir})
            list(APPEND CIRCUIT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        else()
            list(APPEND CIRCUIT_SOURCES "${source}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES CIRCUIT_SOURCES)

    foreach(ITR ${CIRCUIT_SOURCES})
        if(NOT EXISTS ${ITR})
            message(SEND_ERROR "Cannot find circuit source file: ${source}")
        endif()
    endforeach()

    set(INCLUDE_DIRS_LIST "")
    # Collect include directories from dependencies first
    foreach(lib ${ARG_LINK_LIBRARIES})
        get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
        foreach(dir ${lib_include_dirs})
            list(APPEND INCLUDE_DIRS_LIST "-I${dir}")
        endforeach()
    endforeach()
    # Add passed include directories
    foreach(include_dir ${ARG_INCLUDE_DIRECTORIES})
        if(NOT IS_ABSOLUTE ${include_dir})
            set(include_dir "${CMAKE_CURRENT_SOURCE_DIR}/${include_dir}")
        endif()
        list(APPEND INCLUDE_DIRS_LIST "-I${include_dir}")
    endforeach()
    if (ZKLLVM_DEV_ENVIRONMENT)
        list(APPEND INCLUDE_DIRS_LIST -I${CMAKE_SOURCE_DIR}/libs/stdlib/libcpp -I${CMAKE_SOURCE_DIR}/libs/stdlib/libc/include)
    endif()
    list(REMOVE_DUPLICATES INCLUDE_DIRS_LIST)

    if(CIRCUIT_ASSEMBLY_OUTPUT)
        set(extension ll)
        set(format_option -S)
        set(link_options "-S")
    else()
        set(extension bc)
        set(format_option -c)
    endif()

    if (ZKLLVM_DEV_ENVIRONMENT)
        set(CLANG $<TARGET_FILE:clang>)
        set(LINKER $<TARGET_FILE:llvm-link>)
    else()
        set(CLANG clang)
        set(LINKER llvm-link)
    endif()

    # Compile sources
    set(compiler_outputs "")
    add_custom_target(${name}_compile_sources)
    foreach(source ${CIRCUIT_SOURCES})
        get_filename_component(source_base_name ${source} NAME)
        add_custom_target(${name}_${source_base_name}_${extension}
                        COMMAND ${CLANG} -target assigner
                        -D__ZKLLVM__ ${INCLUDE_DIRS_LIST} -emit-llvm -O1 ${format_option} ${ARG_COMPILER_OPTIONS}  -o ${name}_${source_base_name}.${extension} ${source}

                        VERBATIM COMMAND_EXPAND_LISTS

                        SOURCES ${source})
        add_dependencies(${name}_compile_sources ${name}_${source_base_name}_${extension})
        list(APPEND compiler_outputs "${name}_${source_base_name}.${extension}")
    endforeach()

    # Link sources
    add_custom_target(${name}
                      COMMAND ${LINKER} ${link_options} -o ${name}.${extension} ${compiler_outputs}
                      DEPENDS ${name}_compile_sources
                      VERBATIM COMMAND_EXPAND_LISTS)
    if (${ZKLLVM_DEV_ENVIRONMENT})
        add_dependencies(${name} zkllvm-libc)
    endif()
    set_target_properties(${name} PROPERTIES OUTPUT_NAME ${name}.${extension})
endfunction(add_circuit_no_stdlib)

function(add_circuit)
    list(POP_FRONT ARGV circuit_name)
    list(PREPEND ARGV ${circuit_name}_no_stdlib)
    add_circuit_no_stdlib(${ARGV})

    if (ZKLLVM_DEV_ENVIRONMENT)
        set(LINKER $<TARGET_FILE:llvm-link>)
        set(libc_stdlib ${CMAKE_BINARY_DIR}/libs/stdlib/libc/zkllvm-libc.ll)
    else()
        set(LINKER llvm-link)
        set(libc_stdlib "/usr/lib/zkllvm/zkllvm-libc.ll")
    endif()
    set(link_options "-S")

    add_custom_target(${circuit_name}
                      COMMAND ${LINKER} ${link_options} -o ${circuit_name}.ll ${circuit_name}_no_stdlib.ll ${libc_stdlib}
                      DEPENDS ${circuit_name}_no_stdlib
                      VERBATIM COMMAND_EXPAND_LISTS)
endfunction(add_circuit)
