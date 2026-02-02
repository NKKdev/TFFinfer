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
#include "kernel/include/TFFOPCreatorBase.h"
#include "runtime/MemManager.h"
#include "global/GlobalDefine.h"
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
        NodeMetadata(const char *name) : _is_input(false), _is_output(false), _name(name) {
        };
        NodeMetadata(const bool &is_input, const bool &is_output, const std::string &name) : _is_input(is_input),
            _is_output(is_output), _name(name) {
        };

    };

    class GraphNode : public std::enable_shared_from_this<GraphNode> {
        friend class Graph;

    public:
        explicit GraphNode(const std::string &name = ""):
        _is_fused(false),_memory_type(core::memory::MemoryType::TFF_MEM_TYPE_WEIGHT) {
            this->_node_metadata._name = name;
            this->_params_ptr = std::make_shared<tff::core::global::ParamBaseObject>();
            this->_mem_manager_ptr = std::dynamic_pointer_cast<tff::core::runtime::LLMMemManager>(
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                    WEIGHT_MEM_BUFFER_MANAGER_FLAG,
                    tff::factory::ModuleKeyType(WEIGHT_MEM_BUFFER_MANAGER_FLAG)));
        };

        virtual ~GraphNode() = default;

        bool operator==(const GraphNode &node) const {
            return this->_node_metadata._name == node._node_metadata._name;
        }
        bool operator!=(const GraphNode &node) const {
            return this->_node_metadata._name != node._node_metadata._name;
        }
    public:
        //
        virtual std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() {
            const auto dev = device();
            if (dev.empty()) {
                tff::log::Logger::error("Node '%s': no valid device bound", this->_node_metadata._name.c_str());
                return nullptr;
            }
            auto device_id = this->_devices.begin()->first;
            auto device_ptr = this->_devices.begin()->second;
            if (this->_tensor->get_buffer() == nullptr) {
                auto buffer = this->_mem_manager_ptr->allocate_memory(
                    this->_tensor->get_bytes(), device_id);
                this->_tensor->set_buffer_data(buffer.second, this->_tensor->get_bytes(),
                    buffer.first);
                this->_tensor->set_allocator(device_ptr->get_device_buffer_allocator(device_id));
            }
            //
            for (auto &consumer : this->output_nodes()) {
                auto event = consumer->event();
                this->_mem_manager_ptr->aquire_memory(device_id,
                    this->_tensor->get_external_memory_index(), this->_tensor->get_bytes(),
                    event);
            }
            //
            std::vector<std::shared_ptr<core::device::DeviceEvent>> events_list;
            for (auto &input : this->input_nodes()) {
                events_list.push_back(input->event());
                this->add_inputs(input->get_tensor());
            }
            //
            auto params_ptr = this->get_params();
            params_ptr->set_param(this->_inputs);
            params_ptr->set_param(this->_tensor);
            params_ptr->set_param(this->_node_metadata._name);
            params_ptr->set_param(this->_mem_manager_ptr);
            params_ptr->set_param(this->event());
            params_ptr->set_param(events_list);
            params_ptr->set_param(this->stream());

            auto callback = kernel::base::get_op_func(this->_devices.begin()->second,
                this->_op_type, this->data_type());
            return callback;
        };
    public:
        //
        inline std::string name() const{
            return this->_node_metadata._name;
        }
        //
        void set_file_idx(const uint32_t &file_idx) { _file_idx = file_idx; }

        // 设置节点名称
        void set_name(const std::string &name) { this->_node_metadata._name = name; }

        // 获取算子类型
        TffOpType op_type() const { return _op_type; }
        void set_op_type(const TffOpType type) { _op_type = type; }

        // 获取层类型（可选）
        tff::core::model::ModelTensorLayerType layer_type() const { return _layer_type; }
        //
        inline void bind_devices(std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject>> &device) {
            this->_devices = device;
            if (!this->_devices.empty()) {
                auto device_iter = this->_devices.begin();
                this->_device_stream_ptr = device_iter->second->create_stream(device_iter->first);
                if (this->_device_stream_ptr != nullptr) {
                    std::string stream_name = this->name() + "_stream";
                    this->_device_stream_ptr->set_name(stream_name);
                }

                this->_device_event_ptr = device_iter->second->create_event(device_iter->first);
                if (this->_device_event_ptr != nullptr) {
                    std::string event_name = this->name() + "_event";
                    this->_device_event_ptr->set_name(event_name);
                }

            }
        }

        std::unordered_map<int, std::shared_ptr<device::DeviceBaseObject>> device() const {
            return _devices;
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
            *this->_params_ptr = *params;
        }
        //
        inline std::shared_ptr<tff::core::global::ParamBaseObject> get_params() const {
            return this->_params_ptr;
        }

        //
        inline bool is_fuse() const {
            return this->_is_fused;
        }
        //
        inline void fuse() {
            this->_is_fused = true;
        }

        //
        inline tff::core::memory::DataType data_type() const {
            return this->_tensor->get_data_type();
        }
        //
        inline bool is_leaf() const {
            return this->is_input_node();
        }
        inline void set_tensor(const std::shared_ptr<tff::core::memory::Tensor> &tensor) {
            this->_tensor = tensor;
        }
        //
        inline const std::shared_ptr<tff::core::memory::Tensor> &get_tensor() {
            return this->_tensor;
        }
        inline std::shared_ptr<GraphNode> add_input_node(const std::shared_ptr<GraphNode> &src_node) {
            if (src_node ==nullptr) {
                return src_node;
            }
            auto iter = std::find(this->_input_nodes.begin(), this->_input_nodes.end(), src_node);
            if (iter == this->_input_nodes.end()) {
                for (auto &node:this->input_nodes()) {
                    if (node->name() == src_node->name()) {
                        return src_node;
                    }
                }
                if (this->_devices.begin()->first != src_node->device().begin()->first && this->op_type() != TFF_OP_MEM_CPY) {//设备不一致，自动插入拷贝节点;
                    auto mem_cpy_node = ADD_NODE(TFF_OP_MEM_CPY);
                    mem_cpy_node->set_node_meta(NodeMetadata(src_node->name() + "_mem_cpy"));
                    mem_cpy_node->bind_devices(this->_devices);
                    auto tensor = std::make_shared<tff::core::memory::Tensor>(src_node->get_tensor());
                    mem_cpy_node->set_tensor(tensor);
                    auto params = mem_cpy_node->get_params();
                    auto src_type = src_node->device().begin()->second->get_device_type(src_node->device().begin()->first);
                    auto dst_type = this->_devices.begin()->second->get_device_type(src_node->device().begin()->first);
                    params->set_param(make_cpy_kind(src_type, dst_type));
                    mem_cpy_node->add_input_node(src_node);

                    this->_input_nodes.push_back(mem_cpy_node);
                    //this->add_inputs(mem_cpy_node->get_tensor());
                    return mem_cpy_node;
                }else {
                    this->_input_nodes.push_back(src_node);
                    //this->add_inputs(src_node->get_tensor());
                    return src_node;
                }
            }
            return src_node;
        }
        inline void add_output_node(const std::shared_ptr<GraphNode> &node) {
            this->_output_nodes.push_back(node);
        }
        //
        inline void add_inputs( const std::shared_ptr<core::memory::Tensor> &tensor) {
            this->_inputs.push_back(tensor);
        }
        inline void remove_input_node(const std::shared_ptr<GraphNode> &src_node) {
            std::vector<std::shared_ptr<GraphNode>>::iterator iter = this->_input_nodes.begin();
            for (; iter != this->_input_nodes.end();) {
                if (*iter == src_node) {
                    iter = this->_input_nodes.erase(iter);
                }else {
                    ++iter;
                }
            }
        }
        inline std::vector<std::shared_ptr<GraphNode>> input_nodes() const {
            return this->_input_nodes;
        }
        inline std::vector<std::shared_ptr<GraphNode>> output_nodes() const {
            return this->_output_nodes;
        }
        //
        inline std::shared_ptr<core::device::DeviceEvent> event() const {
            return this->_device_event_ptr;
        }
        //
        inline std::shared_ptr<core::device::DeviceStream> stream() const {
            return this->_device_stream_ptr;
        }
        //
        inline void set_mem_type(const tff::core::memory::MemoryType &mem_type) {
            this->_memory_type = mem_type;
        }
        inline tff::core::memory::MemoryType mem_type() const {
            return this->_memory_type;
        }
    protected:
        NodeMetadata _node_metadata;
        bool _is_fused;


        mutable std::mutex _mutex;

        uint32_t _layer_id = 0;
        uint32_t _file_idx = 0;

        TffOpType _op_type = TFF_OP_NONE;
        core::memory::MemoryType _memory_type;
        std::shared_ptr<tff::core::global::ParamBaseObject> _params_ptr;

        tff::core::model::ModelTensorLayerType _layer_type =
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_NONE;

        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject>> _devices;
        //
        std::shared_ptr<tff::core::runtime::LLMMemManager> _mem_manager_ptr;

        //
        std::shared_ptr<core::device::DeviceStream> _device_stream_ptr;
        std::shared_ptr<core::device::DeviceEvent> _device_event_ptr;

        std::vector<std::shared_ptr<GraphNode>> _input_nodes;
        std::vector<std::shared_ptr<GraphNode>> _output_nodes;
        std::vector<std::shared_ptr<memory::Tensor>> _inputs;
        std::shared_ptr<tff::core::memory::Tensor> _tensor;

    };


}


#endif //TFFINFER_GRAPHNODE_H
