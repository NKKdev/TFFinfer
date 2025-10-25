//
// Created by nkk on 2025/9/28.
//
#include <stdio.h>
#include <iostream>
#include "device/cuda/cudaInc.h"
int main(int argc,char *argv[]) {

    char c = ' ';
    if (isspace(c)) {
        std::cout << "Space detected\n";  // 输出
    }

    for (char c : " \t\n\r\v\f") {
        std::cout << "'" << c << "' is space: " << isspace(c) << "\n";
    }

    int device_cnt = 0;
    CudaSafeCall(cudaGetDeviceCount(&device_cnt));
    std::vector<cudaDeviceProp> device_props;
    for (size_t i = 0; i < device_cnt; i++) {
        cudaDeviceProp device_prop{};
        CudaSafeCall(cudaGetDeviceProperties(&device_prop, i));
        device_props.push_back(device_prop);
    }
    return 0;
}