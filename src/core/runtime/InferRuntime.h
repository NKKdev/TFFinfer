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
#include "KVCache.h"
#include "../device/MemBufferAllocatorBaseObject.h"
#include "ExportInc.h"
#include "GraphOptimizer.h"
#include "MemManager.h"
#include "device/DeviceManager.h"
#include "model/base/ModelCreatorBase.h"
#include "runtime/BatchManager.h"
#include "runtime/TaskFlowManager.h"

namespace tff::core::runtime {
    /**
     * @brief 推理运行时类
     */
    class DEEP_TFF_API LLMInferRuntime {
    public:
        LLMInferRuntime() : _type(), _architecture() {
            this->init_device();
            //
            this->_mem_manager_ptr = std::dynamic_pointer_cast<runtime::LLMMemManager>(
                factory::ModuleFactory::instance()
                ->create_shared<tff::module::ModuleObject>(WEIGHT_MEM_BUFFER_MANAGER_FLAG,
                                                           tff::factory::ModuleKeyType(
                                                               WEIGHT_MEM_BUFFER_MANAGER_FLAG)));
            if (this->_mem_manager_ptr != nullptr) {
                this->_mem_manager_ptr->init_device(this->_device_manager->devices());
            }

            this->_llm_batch_manager_ptr = std::dynamic_pointer_cast<tff::core::runtime::LLMBatchManager>(
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                    BATCH_MANAGER_FLAG,
                    tff::factory::ModuleKeyType(BATCH_MANAGER_FLAG)));
            this->_task_manager = std::dynamic_pointer_cast<tff::core::runtime::LLMTaskFlowManager>(
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                    TASK_FLOW_MANAGER_FLAG,
                    tff::factory::ModuleKeyType(tff::core::global::TaskFlowType::TFF_FLOW_LLM)));
            //
            this->_graph_optimizer = std::dynamic_pointer_cast<GraphOptimizer>(
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                    GRAPH_OPTIMIZER_FALG, tff::factory::ModuleKeyType(GRAPH_OPTIMIZER_FALG)));
        }

        ~LLMInferRuntime() {
        }

    public:
        /**
         * @brief 加载模型
         * @param model_files_path 模型文件路径
         * @param params 模型参数
         * @return 是否成功
         */
        bool load_model(const std::vector<std::string> &model_files_path,
                        const tff::core::model::ModelConfig &params);

        /**
         * @brief 加载模型配置文件
         * @param model_config_file_path
         * @param params
         * @return
         */
        bool load_model_config(const std::string &model_config_file_path, tff::core::model::ModelConfig &params);

        /**
         * @brief 构建模型创建器
         * @return 是否成功
         */
        bool build_model_creator();

        /**
         * @brief 初始化设备
         * @return 是否成功
         */
        bool init_device();

        /**
         * @brief 初始化运行时上下文
         * @return 是否成功
         */
        bool init_runtime_context();

        /**
         * @brief 初始化KV缓存
         * @return 是否成功
         */
        bool init_kvcache();

        /**
         * @brief 初始化Graph
         * @param graph_ptr
         * @return
         */
        bool init_graph(std::shared_ptr<graph::Graph> &graph_ptr);

        /**
         * @brief 初始化内存IO计算图
         * @param graph_ptr  图
         * @return
         */
        bool init_io_graph(std::shared_ptr<tff::core::graph::Graph> &graph_ptr);

        /**
         * @brief 初始化内存管理器
         * @param graph_ptr
         * @param _mem_manager_ptr
         * @return
         */
        bool init_mem_manager(const std::shared_ptr<graph::Graph> &graph_ptr,
                              std::shared_ptr<LLMMemManager> &_mem_manager_ptr) const;

        /**
         * @brief 推理
         * @param n_predict 预测字符数
         * @param generate_str 生成的字符串
         * @return
         */
        bool infer(int n_predict, std::vector<std::string> &generate_str);

        /**
         * @brief 预填充
         * @param ubatch
         * @return
         */
        bool prefill(std::shared_ptr<LLMBatch> &ubatch);

        /**
         * @brief 解码
         * @param ubatch
         * @param n_predict
         * @param generate_str
         * @return
         */
        bool decode(std::shared_ptr<LLMBatch> &ubatch, const int &n_predict, std::string &generate_str);

        /**
         * @brief 编码prompt
         * @param prompt_batches
         * @return
         */
        int encode(const std::vector<std::string> &prompt_batches);

        /**
         * @brief 构建输入
         * @param batch
         */
        void build_inputs(std::shared_ptr<LLMBatch> &batch);

        /**
         * @brief 构建输出
         */
        void build_output();

        /**
         * @brief 更新KV缓存
         * @param graph_ctx
         * @param ubatch
         */
        void update_kv_cache(model::GraphContext &graph_ctx, std::shared_ptr<LLMBatch> &ubatch);

    protected:
        /**
         * @brief 加载架构信息
         */
        inline void load_arch() {
            this->_arch_name = tff::core::global::LLM_ARCH_NAMES.find(this->_architecture)->second;
        }

        /**
         * @brief 加载参数
         */
        void load_hparams(bool is_fuse_op, const memory::DataType &kv_data_type);

        /**
         * @brief 加载词表
         */
        void load_vocab() const;

        /**
         * @brief 加载权重
         */
        bool build_layers();

        /**
         * @brief 加载权重数据
         */
        bool load_tensor_data();

        /**
         * @brief 绑定设备
         */
        void bind_device(std::shared_ptr<layer::ModelLayerObject> &layer_obj, const int &total_layer_index);

        /**
         * @brief 构建内存偏移(已废弃)
         */
        void build_mem_offset(const std::shared_ptr<LLMMemManager> &_mem_manager_ptr,
                              const std::shared_ptr<graph::Graph> &graph_ptr,
                              std::unordered_map<std::string, std::unordered_map<int, size_t> > &
                              mem_buffer_offset_map) const;

        /**
         * @brief 采样token
         * @return  token
         */
        int32_t sample_token();

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
        std::shared_ptr<tff::core::model::LLMVocabulary> _vocabulary_ptr;

        std::unordered_map<std::string, std::string> _model_meta_kv;


        std::shared_ptr<tff::core::model::ModelLoaderBase> _model_loader;
        std::shared_ptr<tff::core::model::ModelCreatorBase> _model_creator;
        //
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
            std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::model::layer::ModelLayerObject> > > >
        _layer_map;
        //
        std::shared_ptr<tff::core::graph::Graph> _prefill_graph_ptr;
        //
        std::shared_ptr<tff::core::graph::Graph> _decode_graph_ptr;

        std::shared_ptr<tff::core::graph::Graph> _mem_graph_ptr;
        //
        std::shared_ptr<GraphOptimizer> _graph_optimizer;

        std::unordered_map<int, std::shared_ptr<LLMKVCache> > _kv_cache_ptr;

        std::shared_ptr<tff::core::runtime::LLMBatchManager> _llm_batch_manager_ptr;
        //
        std::shared_ptr<tff::core::runtime::LLMMemManager> _mem_manager_ptr;
        //
        std::shared_ptr<tff::core::runtime::LLMTaskFlowManager> _task_manager;
        //
        std::shared_ptr<tff::core::device::DeviceManager> _device_manager;
    };
}
#endif //TFFINFER_LLMRUNTIME_H
