//
// Created by nkk on 2025/10/24.
//

#ifndef TFFINFER_GRAPHNODE_H
#define TFFINFER_GRAPHNODE_H
#include <memory>
#include "BaseDefine.h"
#include "model/BaseDefine.h"
#include "mem/Tensor.h"
#include "global/GlobalDefine.h"
#include "device/DeviceBaseObject.h"
namespace tff::core::graph {
    class GraphNode final :public std::enable_shared_from_this<GraphNode>{
    public:
        GraphNode() = default;

        virtual ~GraphNode() = default;

    public:
        std::vector<std::shared_ptr<tff::core::memory::Tensor>> _src_tensors_ptr;
        std::vector<std::shared_ptr<tff::core::memory::Tensor>> _dst_tensors_ptr;
        //
        TffOpType _op_type;
        //
        int32_t _op_params[TFF_MAX_OP_PARAMS / sizeof(int32_t)];
        //
        tff::core::model::ModelTensorLayerType _layer_type;
        //
        std::unordered_map<uint32_t,std::shared_ptr<tff::core::device::DeviceBaseObject>> _devices_list;
    protected:

    };
}


#endif //TFFINFER_GRAPHNODE_H
