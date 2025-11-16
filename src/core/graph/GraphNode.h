//
// Created by nkk on 2025/10/24.
//

#ifndef TFFINFER_GRAPHNODE_H
#define TFFINFER_GRAPHNODE_H
#include <memory>
#include <set>
#include <utility>
#include "BaseDefine.h"
#include "model/BaseDefine.h"
#include "mem/Tensor.h"
#include "global/ParamBaseObject.h"
#include "device/DeviceBaseObject.h"
#include "Logger.h"
class Graph;

namespace tff::core::graph {
    //
    enum GraphNodeType {
        TFF_GRAPH_NODE_UNKNOWN,
        TFF_GRAPH_NODE_COMPUTE,
        TFF_GRAPH_NODE_MAP2CPU,
        TFF_GRAPH_NODE_CPU2GPU,
        TFF_GRAPH_NODE_GPU2CPU,
        TFF_GRAPH_NODE_W,
        TFF_GRAPH_NODE_B,
    };

    struct NodeMetadata {
        bool _is_input{};
        bool _is_output{};
        std::string _name;

        NodeMetadata &operator=(const NodeMetadata &meta) = default;

        bool operator==(const NodeMetadata &meta) const {
            return _is_input == meta._is_input && _is_output == meta._is_output && _name == meta._name;
        }

        NodeMetadata() : _is_input(false), _is_output(false), _name() {
        };
        explicit NodeMetadata(std::string name) : _is_input(false), _is_output(false), _name(std::move(name)) {
        };
        NodeMetadata(const bool &is_input, const bool &is_output, const std::string &name) : _is_input(is_input),
            _is_output(is_output), _name(name) {
        };

    };

    class GraphNode : public std::enable_shared_from_this<GraphNode> {
        friend class Graph;

    public:
        explicit GraphNode(const std::string &name = "") {
            this->_node_metadata._name = name;
        };

        virtual ~GraphNode() = default;

        // 获取节点名称
        const std::string &name() const { return this->_node_metadata._name; }
        //
        void set_file_idx(const uint32_t &file_idx) { _file_idx = file_idx; }

        // 设置节点名称
        void set_name(const std::string &name) { this->_node_metadata._name = name; }

        // 获取算子类型
        TffOpType op_type() const { return _op_type; }

        void set_op_type(const TffOpType type) { _op_type = type; }

        // 获取层类型（可选）
        tff::core::model::ModelTensorLayerType layer_type() const { return _layer_type; }

        // 输入/输出 Tensor
        const std::vector<std::shared_ptr<tff::core::memory::Tensor> > &inputs() const { return _src_tensors_ptr; }
        const std::vector<std::shared_ptr<tff::core::memory::Tensor> > &outputs() const { return _dst_tensors_ptr; }

        void set_inputs(const std::vector<std::shared_ptr<tff::core::memory::Tensor> > &inputs) {
            _src_tensors_ptr.reserve(inputs.size());
            _src_tensors_ptr.insert(_src_tensors_ptr.end(), inputs.begin(), inputs.end());
        }

        void set_outputs(const std::vector<std::shared_ptr<tff::core::memory::Tensor> > &outputs) {
            _dst_tensors_ptr.reserve(outputs.size());
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
                case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING: {
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


        std::shared_ptr<tff::core::global::ParamBaseObject> op_params() { return this->_params_ptr; }
        std::shared_ptr<tff::core::global::ParamBaseObject> op_params() const { return this->_params_ptr; }

        const auto &devices() const { return _devices; }
        //
        inline void bind_devices(const std::shared_ptr<tff::core::device::DeviceBaseObject> &device) {
            this->_devices.insert(device);
        }

        std::shared_ptr<tff::core::device::DeviceBaseObject> device() const {
            return _devices.empty() ? nullptr : *_devices.begin();
        }
        //
        virtual std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() {
            return nullptr;
        } ;
        //
        virtual void release() {
        }
        //
        inline void set_node_meta(const NodeMetadata &meta) {
            this->_node_metadata = meta;
        };

        inline bool is_input_node() const {
            return this->_node_metadata._is_input;
        }

        //
        inline bool is_output_node() const {
            return this->_node_metadata._is_output;
        }
        //
        inline void set_params(const std::shared_ptr<tff::core::global::ParamBaseObject> &params) {
            this->_params_ptr = params;
        }
        //
        inline std::shared_ptr<tff::core::global::ParamBaseObject> get_params() const {
            return this->_params_ptr;
        }
        //
        inline std::vector<std::weak_ptr<GraphNode>> get_predecessors() const {
            return this->_prev_nodes;
        }
        //
        inline std::vector<std::weak_ptr<GraphNode> > get_successors() const {
            return this->_next_nodes;
        }

    protected:
        NodeMetadata _node_metadata;

        std::vector<std::weak_ptr<GraphNode> > _prev_nodes; // 前驱节点
        std::vector<std::weak_ptr<GraphNode> > _next_nodes; // 后继节点
        mutable std::mutex _mutex;

        uint32_t _layer_id = 0;
        uint32_t _file_idx = 0;

        std::vector<std::shared_ptr<tff::core::memory::Tensor> > _src_tensors_ptr;
        std::vector<std::shared_ptr<tff::core::memory::Tensor> > _dst_tensors_ptr;

        TffOpType _op_type = TFF_OP_NONE;
        std::shared_ptr<tff::core::global::ParamBaseObject> _params_ptr;

        tff::core::model::ModelTensorLayerType _layer_type =
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_NONE;

        std::set<std::shared_ptr<tff::core::device::DeviceBaseObject>, tff::core::device::DevicePtrComparator> _devices;
    };

    struct CompareByLayerID {
        bool operator()(const std::shared_ptr<GraphNode> &node1, const std::shared_ptr<GraphNode> &node2) const {
            if (node1->layer_id() != node2->layer_id()) {
                return node1->layer_id() < node2->layer_id();
            }
            return node1->name() < node2->name();
        }
    };
}


#endif //TFFINFER_GRAPHNODE_H
