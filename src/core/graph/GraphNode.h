//
// Created by nkk on 2025/10/24.
//

#ifndef TFFINFER_GRAPHNODE_H
#define TFFINFER_GRAPHNODE_H
#include <memory>
#include <unordered_set>
#include <functional>
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
#include "include/Builder.h"


namespace tff::kernel::builder {
    class IParamBuilder;
}

class Graph;

namespace tff::core::graph {
    /**
     * @brief 节点类型
     */
    enum GraphNodeType {
        TFF_GRAPH_NODE_UNKNOWN,
        TFF_GRAPH_NODE_COMPUTE,
        TFF_GRAPH_NODE_MAP2CPU,
        TFF_GRAPH_NODE_CPU2GPU,
        TFF_GRAPH_NODE_GPU2CPU,
        TFF_GRAPH_NODE_W,
        TFF_GRAPH_NODE_B,
    };

    /**
     * @brief 节点元数据
     */
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

    /**
     * @brief 节点类
     */
    class GraphNode : public std::enable_shared_from_this<GraphNode> {
        friend class Graph;

    public:
        explicit GraphNode(const std::string &name = "") : _is_fuse(false) {
            this->_node_metadata._name = name;
            //this->_params_ptr = std::make_shared<tff::core::global::ParamBaseObject>();
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
        /**
         * @brief 参数检查
         */
        virtual void prepare_params() = 0;

        /**
         * @brief 形状推导
         */
        virtual void shape_infer() = 0;

    public:
        /**
         * @brief 前向计算 负责静态计算形状推导以及显存分配等，真正的计算在调度器中完成
         * @return std::function<tff::kernel::base::OP_CALLBACK_TYPE> 用于调度器计算的回调函数
         */
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward();

    public:
        /**
         * @brief 获取节点名称
         * @return std::string
         */
        inline std::string name() const {
            return this->_node_metadata._name;
        }

        /**
         * @brief 设置节点名称
         * @param name
         */
        void set_name(const std::string &name) { this->_node_metadata._name = name; }
        /**
         * @brief 获取节点OP类型
         * @return TffOpType
         */
        TffOpType op_type() const { return _op_type; }
        /**
         * @brief 设置节点OP类型
         * @param type
         */
        void set_op_type(const TffOpType type) { _op_type = type; }

        /**
         * @brief 绑定设备
         * @param device   设备列表
         */
        inline void bind_devices(
            const std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > &device) {
            if (!device.empty()) {
                auto device_iter = device.begin();
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

        /**
         * @brief 设置节点元数据
         * @return NodeMetadata
         */
        inline void set_node_meta(const NodeMetadata &meta) {
            this->_node_metadata = meta;
        };
        /**
         * @brief 获取节点元数据
         * @return bool 是否是输入节点
         */
        inline bool is_input_node() const {
            return this->_node_metadata._is_input;
        }

        /**
         * @brief 获取节点元数据
         * @return bool 是否是输出节点
         */
        inline bool is_output_node() const {
            return this->_node_metadata._is_output;
        }

        /**
         * @brief 获取节点数据类型
         * @return tff::core::memory::DataType
         */
        inline tff::core::memory::DataType data_type() const {
            if (this->_tensor == nullptr) {
                return memory::TFF_DATA_TYPE_F32;
            }
            return this->_tensor->get_data_type();
        }

        /**
         * @brief 是否是叶子节点
         * @return  bool
         */
        inline bool is_leaf() const {
            return this->_input_nodes.empty();
        }

        /**
        * @brief 设置节点数据
        * @param  tensor
        */
        inline void set_tensor(const std::shared_ptr<tff::core::memory::Tensor> &tensor) {
            this->_tensor = tensor;
        }

        /**
         * @brief 获取节点数据
         * @return std::shared_ptr<tff::core::memory::Tensor>
         */
        inline const std::shared_ptr<tff::core::memory::Tensor> &get_tensor() {
            return this->_tensor;
        }

        /**
         * @brief 添加输入节点
         * @param src_node
         */
        void add_input_node(const std::shared_ptr<GraphNode> &src_node);

        /**
         * @brief 添加输出节点
         * @param node
         */
        inline void add_output_node(const std::shared_ptr<GraphNode> &node) {
            if (node == nullptr) {
                return;
            }

            this->_output_nodes.push_back(node);
        }

        /**
         * @brief 添加输入节点
         * @param src_node
         */
        inline void add_inputs(const std::shared_ptr<core::memory::Tensor> &tensor) {
            auto iter = std::find(this->_inputs.begin(), this->_inputs.end(), tensor);
            if (iter == this->_inputs.end()) {
                this->_inputs.push_back(tensor);
            }
        }

        /**
         * @brief 删除输入节点
         * @param src_node
         */
        inline void remove_input_node(const std::shared_ptr<GraphNode> &src_node) {
            std::vector<std::shared_ptr<GraphNode> >::iterator iter = this->_input_nodes.begin();
            for (; iter != this->_input_nodes.end();) {
                if (*iter == src_node) {
                    iter = this->_input_nodes.erase(iter);
                } else {
                    ++iter;
                }
            }
        }

        /**
         * @brief 删除输出节点
         * @param node
         */
        inline void remove_output_node(const std::shared_ptr<GraphNode> &node) {
            auto iter = std::find(this->_output_nodes.begin(), this->_output_nodes.end(), node);
            if (iter != this->_output_nodes.end()) {
                this->_output_nodes.erase(iter);
            }
        }

        /**
         * @brief 获取当前节点的输入节点
         * @return  std::vector<std::shared_ptr<GraphNode> >
         */
        inline std::vector<std::shared_ptr<GraphNode> > input_nodes() const {
            return this->_input_nodes;
        }

        /**
         * @brief 获取当前节点的输出节点
         * @return  std::vector<std::shared_ptr<GraphNode> >
         */
        inline std::vector<std::shared_ptr<GraphNode> > output_nodes() const {
            return this->_output_nodes;
        }

        /**
         * @brief 获取当前节点绑定的设备事件
         * @return  std::shared_ptr<device::DeviceEvent> 绑定的当前设备事件
         */
        inline const std::shared_ptr<device::DeviceEvent> &event() const {
            return this->_device_event_ptr;
        }

        /**
         * @brief 获取当前节点绑定的设备流
         * @return  std::shared_ptr<device::DeviceEvent> 返回当前节点绑定的设备流
         */
        inline std::shared_ptr<core::device::DeviceStream> stream() const {
            return this->_device_stream_ptr;
        }

        /**
         * @brief 设置层ID
         * @param layer_id
         */
        inline void set_layer_id(const int &layer_id) {
            // tff::log::Logger::info("GraphNode::set_layer_id layer_id: {%d} to %d",
            //                        this->_layer_id, layer_id);
            this->_layer_id = layer_id;
        }

        /**
         * @brief 获取层ID
         * @return int
         */
        inline int layer_id() const {
            return this->_layer_id;
        }

        /**
         * @brief 设置参数构建器
         * @param builder
         */
        inline void set_builder(const std::shared_ptr<tff::kernel::builder::IParamBuilder> &builder) {
            this->_builder = builder;
        }

        /**
         * @brief 获取参数构建器
         * @return std::shared_ptr<kernel::builder::IParamBuilder>
         */
        inline std::shared_ptr<kernel::builder::IParamBuilder> &get_builder() {
            return this->_builder;
        }

        /**
         * @brief 获取参数
         * @return std::shared_ptr<core::global::ParamBaseObject>
         */
        inline std::shared_ptr<core::global::ParamBaseObject> get_params() {
            return this->_builder->build();
        }

        /**
         * @brief 是否融合
         */
        inline bool is_fuse() {
            return this->_is_fuse;
        }

        /**
         * @brief 融合节点
         */
        inline void fuse(const std::shared_ptr<graph::GraphNode> &current_node, bool bOutOverride = true) {
            if (bOutOverride) {
                kernel::builder::IParamBuilder::with_out(this->builder(), current_node->builder());
                const auto &params = current_node->builder()->extract_params();
                const std::shared_ptr<memory::Tensor> &val =
                        std::any_cast<const std::shared_ptr<memory::Tensor> &>(
                            params.find(kernel::builder::IParamBuilder::CommonParams::Out)->second);
                current_node->set_tensor(val);
            } else {
                kernel::builder::IParamBuilder::with(this->builder(), current_node->builder());
            }
            this->_is_fuse = true;
            //current_node->fuse();
        }

        /**
         * @brief 融合节点
         */
        inline void fuse() {
            this->_is_fuse = true;
        }

        /**
         * @brief 获取参数构建器
         * @return std::shared_ptr<tff::kernel::builder::IParamBuilder>
         */
        inline std::shared_ptr<tff::kernel::builder::IParamBuilder> builder() {
            return this->_builder;
        }

        /**
         * @brief 获取张量参数对应的参数名称，用于算子融合
         * @param tensor 参数张量
         * @return const char*
         */
        inline const char *para_name(const std::shared_ptr<memory::Tensor> &tensor) {
            return this->_tensor_param_map[tensor];
        }

    protected:
        NodeMetadata _node_metadata;
        mutable std::mutex _mutex;

        int _layer_id = 0;
        bool _is_fuse;

        TffOpType _op_type = TFF_OP_NONE;

        //
        std::shared_ptr<tff::core::runtime::LLMMemManager> _mem_manager_ptr;

        //
        std::shared_ptr<core::device::DeviceStream> _device_stream_ptr;
        std::shared_ptr<core::device::DeviceEvent> _device_event_ptr;

        std::vector<std::shared_ptr<GraphNode> > _input_nodes;
        std::vector<std::shared_ptr<GraphNode> > _output_nodes;
        std::vector<std::shared_ptr<memory::Tensor> > _inputs;
        std::shared_ptr<tff::core::memory::Tensor> _tensor;

        //
        std::shared_ptr<tff::kernel::builder::IParamBuilder> _builder;
        //
        std::unordered_map<std::shared_ptr<memory::Tensor>, const char *> _tensor_param_map;
    };
    /**
     * @brief 节点语义信息
     */
    struct GraphNodeSemanticKey {
        TffOpType _op_type;
        std::string _params_fingerprint;

        explicit GraphNodeSemanticKey(const std::shared_ptr<tff::core::graph::GraphNode> &node)
            : _op_type(node->op_type()) {
            auto params = node->get_builder()->extract_params();

            std::vector<std::string> kv_pairs;
            for (const auto &[key, value]: params) {
                if (key == "out") {
                    continue;
                }
                std::string val_str;
                std::stringstream ss;
                if (value.type() == typeid(std::shared_ptr<tff::core::memory::Tensor>)) {
                    auto tensor = std::any_cast<std::shared_ptr<tff::core::memory::Tensor> >(value);
                    ss << tensor.get();
                    val_str = "Tensor@" + ss.str();
                } else if (value.type() == typeid(int)) {
                    ss << std::any_cast<int>(value);
                    val_str = ss.str();
                } else if (value.type() == typeid(float)) {
                    ss << std::any_cast<float>(value);
                    val_str = ss.str();
                } else if (value.type() == typeid(std::string)) {
                    val_str = std::any_cast<std::string>(value);
                } else if (value.type() == typeid(bool)) {
                    val_str = std::any_cast<bool>(value);
                } else if (value.type() == typeid(int64_t)) {
                    val_str = std::to_string(std::any_cast<int64_t>(value));
                }
                kv_pairs.push_back(key + "=" + val_str);
            }

            std::sort(kv_pairs.begin(), kv_pairs.end());


            for (const auto &kv: kv_pairs) {
                _params_fingerprint += kv + ";";
            }
        }

        bool operator==(const GraphNodeSemanticKey &other) const {
            return _op_type == other._op_type &&
                   _params_fingerprint == other._params_fingerprint;
        }
    };
}
/**
 * @brief 节点语义信息哈希
 */
template<>
struct std::hash<tff::core::graph::GraphNodeSemanticKey> {
    size_t operator()(const tff::core::graph::GraphNodeSemanticKey &k) const {
        size_t h1 = std::hash<tff::core::graph::TffOpType>()(k._op_type);
        size_t h2 = std::hash<std::string>()(k._params_fingerprint);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

#endif //TFFINFER_GRAPHNODE_H
