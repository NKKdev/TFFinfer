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
#include "mem/Memory.h"
#include "mem/Tensor.h"
#include "model/layer/ModelLayer.h"
#include "model/base/ModelLoaderBase.h"

namespace tff::core::model {
    using NodeType = std::unordered_map<tff::core::graph::GraphNodeType, std::shared_ptr<graph::GraphNode> >;
    struct GraphContext {

        int _n_layer;
        int _n_rot;
        int _n_embd_head;
        int _n_embd_head_k;
        int _n_embd_head_v;
        int _n_head;
        int _n_head_kv;
        int _n_tokens;
        int _max_seq_len;

        float _rope_freq_base;
        float _rope_freq_scale;

        float _f_norm_rms_eps;

        bool _use_fp16;
        bool _use_mmap;

        std::shared_ptr<tff::core::model::ModelLoaderBase> _model_loader;
        std::shared_ptr<tff::core::runtime::LLMMemManager> _mem_manager_ptr;
        GraphContext &operator=(const GraphContext &) = default;
    };
    class ModelCreatorBase {
    public:
        ModelCreatorBase() = default;

        virtual ~ModelCreatorBase() = default;
    public:
        virtual void build_graph(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::model::layer::ModelLayerObject> > > > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr) = 0;

        virtual void build_mem_graph(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::model::layer::ModelLayerObject> > > > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr) = 0;

        //
        virtual const char *get_model_name() = 0;

        //
        virtual void build_model_context(const model::GraphContext &ctx) = 0;
    public:
        model::GraphContext _model_ctx;
    };
}
#endif //TFFINFER_MODELCREATORBASE_H
