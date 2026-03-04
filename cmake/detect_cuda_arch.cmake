if(CMAKE_CUDA_COMPILER AND NOT CMAKE_CUDA_ARCHITECTURES)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        message(STATUS "Detecting NVIDIA GPU architecture using nvidia-smi...")

        execute_process(
                COMMAND ${NVIDIA_SMI} --query-gpu=compute_capability --format=csv=noheader
                OUTPUT_VARIABLE DETECTED_COMPUTE_CAPS
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(DETECTED_COMPUTE_CAPS)
            message("-- DETECTED_COMPUTE_CAPS " ${DETECTED_COMPUTE_CAPS})
            string(REPLACE "." ";" CAP_LIST ${DETECTED_COMPUTE_CAPS})
            list(REMOVE_DUPLICATES CAP_LIST)

            set(CUDA_ARCH_LIST "")
            foreach(cap ${CAP_LIST})
                if(cap MATCHES "^[0-9]+$")
                    list(APPEND CUDA_ARCH_LIST ${cap})
                endif()
            endforeach()

            if(CUDA_ARCH_LIST)
                list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
                list(SORT CUDA_ARCH_LIST)
                string(JOIN ";" CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
                set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "Detected CUDA Architectures" FORCE)
                message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
            else()
                message(WARNING "Could not parse compute capability from nvidia-smi output")
            endif()
        else()
            message(WARNING "nvidia-smi returned no compute capability info")
        endif()
    else()
        message(WARNING "nvidia-smi not found. Please set CMAKE_CUDA_ARCHITECTURES manually.")
    endif()
endif()