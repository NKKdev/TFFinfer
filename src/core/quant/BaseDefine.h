//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_BASEDEFINE_H
#define TFFINFER_BASEDEFINE_H
#include "device/cuda/cudaInc.h"
namespace tff::core::quant {
#define QK8_0 32
    struct Q_8_0{
        half d;       //scale
        int8_t  qs[QK8_0]; // quants
    } ;
}
#endif //TFFINFER_BASEDEFINE_H