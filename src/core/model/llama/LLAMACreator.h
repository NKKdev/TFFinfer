//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_LLAMACREATOR_H
#define TFFINFER_LLAMACREATOR_H
#include "model/base/ModelCreatorBase.h"
#include <memory>

#include "mem/MemBufferAllocatorBaseObject.h"
#include "graph/GraphNode.h"
#include "model/ModelGlobalVar.h"
#include "global/GlobalDefine.h"
#include "FunctionFactory.h"
namespace tff::core::model {
    class LLAMACreator : public ModelCreatorBase<LLAMACreator> {
    public:
        static void create_layer(std::shared_ptr<tff::core::memory::Tensor> &tensor_ptr,
                                      std::shared_ptr<tff::core::graph::GraphNode> &layer_node,
                                      const size_t &total_layer_num = -1, const size_t &layer_index = -1) {
            auto &layer_info = LLM_LAYER_OP_INFOS.find(tensor_ptr->get_tensor_type())->second;
            layer_node = std::make_shared<tff::core::graph::GraphNode>();
            layer_node->_layer_type = layer_info.first;
            layer_node->_op_type = layer_info.second;
            layer_node->_src_tensors_ptr.push_back(tensor_ptr);
            switch (layer_info.first) {
                case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT: {
                    auto device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
                        DEVICE_BACKEND_FLAG, DEVICE_BACKEND_TYPE_CPU);
                    layer_node->_devices_list.insert(std::make_pair(0,device));
                    break;
                }
                case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT: {
                    auto device_cuda = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
                        DEVICE_BACKEND_FLAG, DEVICE_BACKEND_TYPE_CUDA);
                    layer_node->_devices_list.insert(std::make_pair(0,device_cuda));

                    break;
                }
                case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING: {
                    auto device_size = global::get_device_size(DEVICE_BACKEND_TYPE_CUDA);
                    std::vector<float> device_splits;
                    auto device_cuda = tff::factory::ModuleFactory::instance()->create_shared<
                        tff::core::device::DeviceBaseObject>(
                        DEVICE_BACKEND_FLAG, DEVICE_BACKEND_TYPE_CUDA);


                    std::vector<int> device_list;
                    device_cuda->get_device_id(device_list);
                    for (size_t i = 0; i < device_list.size(); ++i) {
                        size_t total_mem;
                        size_t free_mem;
                        device_cuda->get_device_mem(i, &free_mem, &total_mem);
                        device_splits.push_back(static_cast<float>(free_mem));
                    }

                    //计算切分点;
                    float split_sum = 0.0f;
                    for (size_t i = 0; i < device_size; ++i) {
                        split_sum += device_splits[i];
                        device_splits[i] = split_sum;
                    }
                    for (size_t i = 0; i < device_size; ++i) {
                        device_splits[i] /= split_sum;
                    }
                    //
                    const int layer_gpu = std::upper_bound(device_splits.begin(), device_splits.begin() + device_size,
                                                           float(layer_index) / total_layer_num) - device_splits.
                                          begin();
                    layer_node->_devices_list.insert(std::make_pair(layer_gpu,device_cuda));
                    break;
                }
            }
        }
        //
        static const char *get_model_name() {
            return tff::core::model::LLM_ARCH_NAMES.find(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA)->second;
        }
    };

}
#endif //TFFINFER_LLAMACREATOR_H
