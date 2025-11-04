//
// Created by nkk on 2025/10/24.
//

#ifndef TFFINFER_GRAPHNODE_H
#define TFFINFER_GRAPHNODE_H
#include <memory>
#include <set>
#include "BaseDefine.h"
#include "model/BaseDefine.h"
#include "mem/Tensor.h"
#include "global/GlobalDefine.h"
#include "device/DeviceBaseObject.h"
#include "Logger.h"
class Graph;

namespace tff::core::graph {
    class GraphNode : public std::enable_shared_from_this<GraphNode> {
        friend class Graph; // 允许 Graph 修改依赖关系

    public:
        explicit GraphNode(const std::string &name = "") {
            this->_name = name;
        };

        virtual ~GraphNode() = default;

        // 获取节点名称
        const std::string &name() const { return _name; }
        //
        void set_file_idx(const uint32_t &file_idx) { _file_idx = file_idx; }

        // 设置节点名称
        void set_name(const std::string &name) { _name = name; }

        // 获取算子类型
        TffOpType op_type() const { return _op_type; }

        void set_op_type(const TffOpType type) { _op_type = type; }

        // 获取层类型（可选）
        tff::core::model::ModelTensorLayerType layer_type() const { return _layer_type; }

        // 输入/输出 Tensor
        const std::vector<std::shared_ptr<tff::core::memory::Tensor> > &inputs() const { return _src_tensors_ptr; }
        const std::vector<std::shared_ptr<tff::core::memory::Tensor> > &outputs() const { return _dst_tensors_ptr; }

        void set_inputs(std::vector<std::shared_ptr<tff::core::memory::Tensor> > inputs) {
            _src_tensors_ptr.insert(_src_tensors_ptr.end(), inputs.begin(), inputs.end());
        }

        void set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor> > outputs) {
            _dst_tensors_ptr.insert(_dst_tensors_ptr.end(), outputs.begin(), outputs.end());
        }

        //
        inline void set_layer_id(const uint32_t &layer_id) { _layer_id = layer_id; }
        //
        inline uint32_t layer_id() const { return _layer_id; }
        //
        inline void set_layer_type(const tff::core::model::ModelTensorLayerType _layer_type) {
            this->_layer_type = _layer_type;
        }

        //
        inline tff::core::memory::DataType data_type() const {
            tff::core::memory::DataType data_type = memory::DataType::TFF_DATA_TYPE_UNKNOWN;
            switch (_layer_type) {
                case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT:
                case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT:
                case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING:{
                    if (!_src_tensors_ptr.empty()) {
                        data_type = _src_tensors_ptr[0]->get_data_type();
                    }
                    break;
                }
                case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_NONE:
                default:
                    data_type = memory::DataType::TFF_DATA_TYPE_UNKNOWN;
            }
            return data_type;
        }

        // 算子参数（通用）
        int32_t *op_params() { return _op_params; }
        const int32_t *op_params() const { return _op_params; }

        // 设备列表（支持多设备）
        const auto &devices() const { return _devices; }
        //
        inline void bind_devices(const std::shared_ptr<tff::core::device::DeviceBaseObject> &device) {
            this->_devices.insert(device);
        }

        // 获取执行设备（默认取第一个）
        std::shared_ptr<tff::core::device::DeviceBaseObject> device() const {
            return _devices.empty() ? nullptr : *_devices.begin();
        }

        virtual inline bool init() {
            if (this->_op_type == TFF_OP_NONE) {
                tff::log::Logger::error("current op type is invalid: %d\n", this->_op_type);
                return false;
            }
            if (this->_src_tensors_ptr.empty()) {
                tff::log::Logger::error("src_tensors is empty\n");
                return false;
            }
            if (_devices.empty()) {
                tff::log::Logger::error("device is empty\n");
                return false;
            }
            return true;
        }

        //
        virtual void release() {
        }

        //
        virtual bool forward() {
            std::lock_guard<std::mutex> lock(_mutex);

            // 1. 检查是否已初始化必要字段
            if (_op_type == TFF_OP_NONE) {
                tff::log::Logger::error("Node '%s': op_type is TFF_OP_NONE", _name.c_str());
                return false;
            }

            if (_src_tensors_ptr.empty()) {
                tff::log::Logger::error("Node '%s': no input tensors", _name.c_str());
                return false;
            }

            if (_dst_tensors_ptr.empty()) {
                tff::log::Logger::error("Node '%s': no output tensors", _name.c_str());
                return false;
            }

            auto dev = device();
            if (!dev) {
                tff::log::Logger::error("Node '%s': no valid device bound", _name.c_str());
                return false;
            }

            // 2. 确保所有输入 tensor 已分配内存
            for (size_t i = 0; i < _src_tensors_ptr.size(); ++i) {
                auto &tensor = _src_tensors_ptr[i];
                if (!tensor || !tensor->is_allocated()) {
                    tff::log::Logger::error("Node '%s': input tensor[%zu] is null or not allocated", _name.c_str(), i);
                    return false;
                }
            }

            // 3. 确保所有输出 tensor 已分配内存（应在 init() 中完成）
            for (size_t i = 0; i < _dst_tensors_ptr.size(); ++i) {
                auto &tensor = _dst_tensors_ptr[i];
                if (!tensor || !tensor->is_allocated()) {
                    tff::log::Logger::error("Node '%s': output tensor[%zu] is null or not allocated", _name.c_str(), i);
                    return false;
                }
            }
            return true;
        }

    protected:
        std::string _name;

        // 拓扑信息（由 Graph 管理）
        std::vector<std::weak_ptr<GraphNode> > _prev_nodes; // 前驱节点
        std::vector<std::weak_ptr<GraphNode> > _next_nodes; // 后继节点
        mutable std::mutex _mutex;

        uint32_t _layer_id = 0;
        uint32_t _file_idx = 0;

        std::vector<std::shared_ptr<tff::core::memory::Tensor> > _src_tensors_ptr;
        std::vector<std::shared_ptr<tff::core::memory::Tensor> > _dst_tensors_ptr;

        TffOpType _op_type = TFF_OP_NONE;
        int32_t _op_params[TFF_MAX_OP_PARAMS / sizeof(int32_t)] = {0};

        tff::core::model::ModelTensorLayerType _layer_type =
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_NONE;

        std::set<std::shared_ptr<tff::core::device::DeviceBaseObject>, tff::core::device::DevicePtrComparator> _devices;
    };
    struct CompareByLayerID {
        bool operator()(const std::shared_ptr<GraphNode> &node1, const std::shared_ptr<GraphNode> &node2) const{
            if (node1->layer_id() != node2->layer_id()) {
                return node1->layer_id() < node2->layer_id();
            }
            return node1->name() < node2->name();
        }
    };
}


#endif //TFFINFER_GRAPHNODE_H
