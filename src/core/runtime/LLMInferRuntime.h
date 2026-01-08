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
#include "../global/ModelGlobalVar.h"
#include "graph/GraphNode.h"
#include "taskgraph/include/TaskFlowSchedule.h"
#include "mem/LLMKVCache.h"
#include "mem/MemBufferAllocatorBaseObject.h"
#include "ExportInc.h"
#include "LLMMemManager.h"
#include "model/base/ModelCreatorBase.h"
#include "runtime/LLMBatchManager.h"
#include "runtime/LLMTaskFlowManager.h"

namespace tff::core::runtime {
    class DEEP_TFF_API LLMInferRuntime {
    public:
        LLMInferRuntime() : _type(), _architecture() {
            this->init_device();
            //
            //this->_scheduler = std::make_shared<tff::schedule::HybridScheduler>();
            this->_mem_manager_ptr = std::make_shared<tff::core::runtime::LLMMemManager>();
            if (this->_mem_manager_ptr != nullptr) {
                this->_mem_manager_ptr->init_device(this->_devices_map);
            }
            //
            this->_weight_mem_manager_ptr = std::make_shared<tff::core::runtime::LLMMemManager>();
            if (this->_weight_mem_manager_ptr != nullptr) {
                this->_weight_mem_manager_ptr->init_device(this->_devices_map);
            }
            this->_llm_batch_manager_ptr = std::dynamic_pointer_cast<tff::core::runtime::LLMBatchManager>(
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                    BATCH_MANAGER_FLAG,
                    tff::factory::ModuleKeyType(BATCH_MANAGER_FLAG)));
            this->_task_manager = std::dynamic_pointer_cast<tff::core::runtime::LLMTaskFlowManager>(
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                    TASK_FLOW_MANAGER_FLAG,
                    tff::factory::ModuleKeyType(tff::core::global::TaskFlowType::TFF_FLOW_LLM)));
        }

        ~LLMInferRuntime() {
        }

    public:
        //
        bool load_model(const std::vector<std::string> &model_files_path,
                        const tff::core::model::ModelConfig &params);

        //
        bool load_model_config(const std::string &model_config_file_path, tff::core::model::ModelConfig &params);
        //
        bool build_model_creator();

        //
        bool init_device();

        //
        bool init_runtime_context();

        //
        bool init_graph();
        //
        bool init_io_graph();
        //
        bool init_mem_manager(const std::shared_ptr<graph::Graph> &graph_ptr, std::shared_ptr<LLMMemManager> &_mem_manager_ptr) const;

        //
        bool prefill();

        //
        bool decode(const int &n_predict, std::string &generate_str);
        //
        int encode(const std::vector<std::string> &prompt_batches);
        //
        void build_inputs(std::shared_ptr<LLMBatch> &batch);
        //
        void build_output();

    protected:
        void load_stats();

        inline void load_arch() {
            this->_arch_name = tff::core::global::LLM_ARCH_NAMES.find(this->_architecture)->second;
        }


        void load_hparams(bool is_fuse_op, const memory::DataType &kv_data_type);

        void load_vocab() const;

        bool build_layers();

        //
        bool load_tensor_data();
        //
        void bind_device(std::shared_ptr<layer::ModelLayerObject> &layer_obj, const int &total_layer_index);
        //
        void build_mem_offset(const std::shared_ptr<LLMMemManager> &_mem_manager_ptr,
                              const std::shared_ptr<graph::Graph> &graph_ptr, std::unordered_map<std::string, std::unordered_map<int, int>> &
                              mem_buffer_offset_map) const;

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
        std::shared_ptr<tff::core::model::LLMLLaMaVocabulary> _vocabulary_ptr;

        //std::unordered_map<std::string, std::shared_ptr<tff::core::memory::Tensor> > _tensor_map;


        std::unordered_map<std::string, std::string> _model_meta_kv;

        std::set<std::shared_ptr<tff::core::device::DeviceBaseObject>, tff::core::device::DeviceComparator> _devices;
        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject>> _devices_map;

        std::shared_ptr<tff::core::model::ModelLoaderBase> _model_loader;
        std::shared_ptr<tff::core::model::ModelCreatorBase> _model_creator;
        //
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
            std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<tff::core::model::layer::ModelLayerObject> > > >
        _layer_map;
        //
        std::shared_ptr<tff::core::graph::Graph> _infer_graph_ptr;
        std::shared_ptr<tff::core::graph::Graph> _mem_graph_ptr;

    public:
        //std::shared_ptr<tff::schedule::HybridScheduler> _scheduler;
        //
        std::unordered_map<int, std::shared_ptr<tff::core::memory::LLMKVCache>> _kv_cache_ptr;

    public:
        //
        std::shared_ptr<tff::core::runtime::LLMBatchManager> _llm_batch_manager_ptr;
        //
        std::shared_ptr<tff::core::runtime::LLMMemManager> _mem_manager_ptr;
        std::shared_ptr<tff::core::runtime::LLMMemManager> _weight_mem_manager_ptr;
        //
        std::shared_ptr<tff::core::runtime::LLMTaskFlowManager> _task_manager;
    };
}
#endif //TFFINFER_LLMRUNTIME_H
