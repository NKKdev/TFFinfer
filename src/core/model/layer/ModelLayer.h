//
// Created by nkk on 2026/1/4.
//

#ifndef TFFINFER_MODELLAYER_H
#define TFFINFER_MODELLAYER_H
#include <string>
#include <map>
#include "model/BaseDefine.h"
#include "device/DeviceBaseObject.h"
#include "mem/Tensor.h"
namespace tff::core::model::layer {
    class ModelLayerObject {
    public:
        ModelLayerObject() = default;

        virtual ~ModelLayerObject() = default;

    public:
        std::string _layer_name;
        int _layer_index;
        ModelTensorLayerType _type;
        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject>> _device_list;
        std::shared_ptr<tff::core::memory::Tensor> _tensor;
    };
}


#endif //TFFINFER_MODELLAYER_H
