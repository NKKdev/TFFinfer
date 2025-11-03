//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_KERNEL_BASEDEFINE_H
#define TFFINFER_KERNEL_BASEDEFINE_H
#include "device/BaseDefine.h"
#include "device/DeviceBaseObject.h"
#include "mem/BaseDefine.h"
#include "mem/Memory.h"
namespace tff::kernel::base {

    enum class Layout { kRowMajor = 101, kColMajor = 102 };

    enum class Transpose { kNo = 111, kYes = 112, kConjugate = 113 };

    enum class Triangle { kUpper = 121, kLower = 122 };

    enum class Diagonal { kNonUnit = 131, kUnit = 132 };

    enum class Side { kLeft = 141, kRight = 142 };

    enum class KernelMode { kCrossCorrelation = 151, kConvolution = 152 };

    enum class BufferAccess { kReadOnly, kWriteOnly, kReadWrite, kNotOwned };

}
#endif //TFFINFER_BASEDEFINE_H
