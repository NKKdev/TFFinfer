//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_MODELCREATORBASE_H
#define TFFINFER_MODELCREATORBASE_H
#include <functional>
#include <tuple>
#include "global/GlobalDefine.h"
#include "FunctionFactory.h"
#include "global/FunctionTraits.h"
#include "../../runtime/KVCache.h"
#include "mem/Memory.h"
#include "mem/Tensor.h"
#include "model/layer/ModelLayer.h"
#include "model/base/ModelLoaderBase.h"

namespace tff::core::model {
    using NodeType = std::unordered_map<tff::core::graph::GraphNodeType, std::shared_ptr<graph::GraphNode> >;
    /**
     * @brief 计算图上下文
     */
    struct GraphContext {
        int _n_layer;
        int _n_rot;
        int _n_embd_head;
        int _n_embd_head_k;
        int _n_embd_head_v;
        int _n_head;
        int _n_head_kv;
        int _n_tokens;
        int _pre_token_num = 0;

        int _max_seq_len;
        int _n_output;

        float _rope_freq_base;
        float _rope_freq_scale;

        float _f_norm_rms_eps;
        graph::TFFNormType _norm_type = graph::TFFNormType::TFF_NORM_RMS;

        bool _use_fp16;
        bool _use_mmap;
        bool _is_prefill;
        bool _is_fuse;

        //
        core::model::RopeType _rope_type;

        //
        int _seq_id;


        //
        std::shared_ptr<core::memory::Tensor> _logits;
        std::shared_ptr<core::memory::Tensor> _rope_table;
        std::shared_ptr<core::memory::Tensor> _mask;
        std::shared_ptr<core::memory::Tensor> _pos;

        //
        std::unordered_map<int, int> _layer_device_map;


        std::shared_ptr<tff::core::model::ModelLoaderBase> _model_loader;
        std::shared_ptr<tff::core::runtime::LLMMemManager> _mem_manager_ptr;
        std::unordered_map<int, std::shared_ptr<tff::core::runtime::LLMKVCache> > _kv_cache_ptr;

        GraphContext &operator=(const GraphContext &) = default;
    };

    /**
     * @brief 模型计算图创建器基类
     */
    class ModelCreatorBase {
    public:
        ModelCreatorBase() = default;

        virtual ~ModelCreatorBase() = default;

    public:
        /**
         * @brief 创建计算图
         * @param layer_map 层信息
         * @param graph_ptr 计算图
         */
        virtual void build_graph(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::model::layer::ModelLayerObject> > > > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr) = 0;

        /**
         * @brief 创建IO计算图
         * @param layer_map  层信息
         * @param graph_ptr  计算图
         */
        virtual void build_mem_graph(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::model::layer::ModelLayerObject> > > > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr) = 0;

        /**
         * @brief 获取模型名称
         * @return  模型名称
         */
        virtual const char *get_model_name() = 0;

        /**
         * @brief 创建模型上下文
         * @param ctx 上下文
         */
        virtual void build_model_context(const model::GraphContext &ctx) = 0;

    public:
        /**
         * @brief 创建张量转换节点
         * @param node_type 节点类型
         * @param node_ptr 节点指针
         * @return 节点指针
         */
        std::shared_ptr<graph::GraphNode> build_convert_node(memory::DataType type,
                                                             std::shared_ptr<tff::core::graph::GraphNode> &node);

        /**
         * @brief 创建权重节点
         * @param layer_map 层信息
         */
        void build_weight_node(
            std::unordered_map<ModelTensorLayerType, std::unordered_map<uint32_t, std::unordered_map<
                memory::ModelTensorType,
                std::shared_ptr<layer::ModelLayerObject> > > > &layer_map);

        /**
         * @brief 创建归一化节点
         * @param type 归一化类型
         * @param weight_node 归一化权重节点
         * @param x_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_norm(
            graph::TFFNormType type,
            std::shared_ptr<graph::GraphNode> &weight_node, std::shared_ptr<graph::GraphNode> &x_node);

        /**
         * @brief 创建矩阵乘法节点
         * @param a_node 输入节点
         * @param b_node 输入节点
         * @return 矩阵乘法节点
         */
        std::shared_ptr<graph::GraphNode> build_mul_mat_node(
            std::shared_ptr<graph::GraphNode> &a_node, std::shared_ptr<graph::GraphNode> &b_node);

        /**
         * @brief 创建自注意力节点
         * @param name 名称
         * @param layer_id 层ID
         * @param q_node Q
         * @param k_node K
         * @param v_node V
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_attn(
            const char *name,
            const int &layer_id,
            std::shared_ptr<graph::GraphNode> &q_node, std::shared_ptr<graph::GraphNode> &k_node,
            std::shared_ptr<graph::
                GraphNode> &v_node);

        /**
         * @brief 创建加法节点
         * @param a_node
         * @param b_node
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_add_node(
            std::shared_ptr<graph::GraphNode> &a_node,
            std::shared_ptr<graph::GraphNode> &b_node);

        /**
         * @brief 创建FFN节点
         * @param weight_node 权重节点
         * @param x_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_ffn_up(
            std::shared_ptr<graph::GraphNode> &weight_node, std::shared_ptr<graph::GraphNode> &x_node);

        /**
         * @brief 创建FFN门节点
         * @param weight_node 权重节点
         * @param x_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_ffn_gate(
            std::shared_ptr<graph::GraphNode> &weight_node, std::shared_ptr<graph::GraphNode> &x_node);

        /**
         * @brief 创建FFN下采样节点
         * @param weight_node 权重节点
         * @param x_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_ffn_down(
            std::shared_ptr<graph::GraphNode> &weight_node, std::shared_ptr<graph::GraphNode> &x_node);

        /**
         * @brief 创建FFN节点
         * @param type 激活函数类型
         * @param layer_id 层ID
         * @param x_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_ffn(
            graph::TFFUnaryType type, const int &layer_id, std::shared_ptr<graph::GraphNode> &x_node);

        /**
         * @brief 创建激活函数节点
         * @param type 激活函数类型
         * @param up_node 输入节点
         * @param gate_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_unary_op(
            graph::TFFUnaryType type,
            std::shared_ptr<graph::GraphNode> &up_node, std::shared_ptr<graph::GraphNode> &gate_node);

        /**
         * @brief 创建乘法节点
         * @param a_node 输入节点
         * @param weight_node 权重节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_mul_node(
            std::shared_ptr<graph::GraphNode> &a_node, std::shared_ptr<graph::GraphNode> &weight_node);

        /**
         * @brief 创建reshape节点
         * @param input_node 输入节点
         * @param dim0
         * @param dim1
         * @param dim2
         * @param dim3
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_reshape_node(
            std::shared_ptr<graph::GraphNode> &input_node,
            int dim0, int dim1, int dim2, int dim3);

        /**
         * @brief 创建KV缓存存储节点
         * @param tensor_type 缓存类型
         * @param layer_id 层ID
         * @param kv_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_kv_cache_store_node(
            memory::ModelTensorType tensor_type, const int &layer_id, const std::shared_ptr<graph::GraphNode> &kv_node);

        /**
         * @brief 创建KV缓存加载节点
         * @param tensor_type 缓存类型
         * @param layer_id 层ID
         * @param node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_kv_cache_load_node(
            memory::ModelTensorType tensor_type, const int &layer_id, const std::shared_ptr<graph::GraphNode> &node);

        /**
         * @brief 创建mask节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_mask_node();

        /**
         * @brief 创建host节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_host_node(
            std::shared_ptr<layer::ModelLayerObject> &layer, NodeType &input_node);

        /**
         * @brief 创建device节点
         * @param layer_type 层类型
         * @param layer_id 层ID
         * @param input_node 输入节点
         * @param current_cpu_node 当前CPU节点
         * @param is_input 是否是输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_device_node(
            std::shared_ptr<layer::ModelLayerObject> &layer, NodeType &input_node,
            std::shared_ptr<graph::GraphNode> &current_cpu_node, bool
            is_input = false);

        /**
         * @brief 创建层节点
         * @param tensor_type 层类型
         * @param layer_map 层对象
         * @param input_node 输入节点
         * @param is_input 是否是输入节点
         * @return
         */
        NodeType build_layer_node(memory::ModelTensorType tensor_type,
                                  const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                      tff::core::model::layer::ModelLayerObject> > &layer_map, NodeType &input_node,
                                  bool is_input = false);

        /**
         * @brief 创建内存对齐节点，用于量化类型内存对齐
         * @param input_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_aligned_node(std::shared_ptr<graph::GraphNode> &input_node);

        /**
         * @brief 创建内存拷贝节点
         * @param source_device_id 源设备ID
         * @param dest_device_id 目标设备ID
         * @param node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_mem_cpy_node(const int &source_device_id,
                                                             const int &dest_device_id,
                                                             std::shared_ptr<graph::GraphNode> &node);

    public:
        model::GraphContext _graph_ctx;
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<int,
            std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<graph::GraphNode> > > >
        _weight_node_map;
    };
}
#endif //TFFINFER_MODELCREATORBASE_H
