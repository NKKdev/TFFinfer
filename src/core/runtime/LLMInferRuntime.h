//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_LLMRUNTIME_H
#define TFFINFER_LLMRUNTIME_H
#include <unordered_map>
#include <memory>
#include "model/BaseDefine.h"
#include "model/LLMVocabulary.h"
#include "mem/BaseDefine.h"
#include "device/DeviceBaseObject.h"
#include "model/base/ModelLoaderBase.h"
#include "ModuleFactory.h"
#include "model/ModelGlobalVar.h"
#include "graph/GraphNode.h"
#include "taskgraph/include/TaskFlowSchedule.h"
#include "mem/LLMKVCache.h"
#include "mem/MemBufferAllocatorBaseObject.h"
#include "ExportInc.h"
#include "runtime/LLMBatchManager.h"
namespace tff::core::runtime {
    class DEEP_TFF_API LLMInferRuntime {
    public:
        LLMInferRuntime() : _type(), _architecture() {
            //
            this->_scheduler = std::make_unique<tff::schedule::HybridScheduler>();
        }

        ~LLMInferRuntime() = default;

    public:
        //
        bool load_model(const std::vector<std::string> &model_files_path,
                        const tff::core::model::ModelConfig &params);

        //
        bool load_model_config(const std::string &model_config_file_path, tff::core::model::ModelConfig &params);

        //
        bool init_device();

        //
        bool init_runtime_context();
        //
        bool init_graph();
        //
        bool prefill(const std::string &prompt);
        //
        bool decode(const int &n_predict, std::string &generate_str);

    protected:
        void load_stats();

        inline void load_arch() {
            this->_arch_name = tff::core::model::LLM_ARCH_NAMES.find(this->_architecture)->second;
        }


        void load_hparams();

        void load_vocab() const;

        bool load_layers();

        //
        bool load_tensor_data();

    public:
        std::string _name;
        std::string _arch_name;
        uint64_t _n_elements = 0;
        size_t _n_bytes = 0;
        //
        bool _has_gpu_backend = false;

        tff::core::model::ModelType _type;
        tff::core::model::ModelArchitectureType _architecture;

        tff::core::model::ModelConfig _model_config;
        std::unique_ptr<tff::core::model::LLMLLaMaVocabulary> _vocabulary_ptr;

        //std::unordered_map<std::string, std::shared_ptr<tff::core::memory::Tensor> > _tensor_map;


        std::unordered_map<std::string, std::string> _model_meta_kv;

        std::set<std::shared_ptr<tff::core::device::DeviceBaseObject> ,tff::core::device::DevicePtrComparator> _devices;

        std::shared_ptr<tff::core::model::ModelLoaderBase> _model_loader;
        //
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::vector<std::shared_ptr<
            tff::core::graph::GraphNode> > >
        _layer_map;

    public:
        std::unique_ptr<tff::schedule::HybridScheduler> _scheduler;
        //
        std::unique_ptr<tff::core::memory::LLMKVCache> _kv_cache_ptr;
        //
        std::unique_ptr<tff::core::runtime::LLMBatchManager> _llm_batch_manager_ptr;
    };
}
#endif //TFFINFER_LLMRUNTIME_H
