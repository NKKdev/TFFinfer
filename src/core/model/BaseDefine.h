//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_MODEL_BASEDEFINE_H
#define TFFINFER_MODEL_BASEDEFINE_H
#include <string>
#include <variant>

#include "mem/Tensor.h"
#include "GGUFDef.h"
#include "Logger.h"
#include "global/OPDefine.h"
using namespace tff::core::global;
namespace tff::core::model {
#define TFF_TENSOR_MAX_DIMS 4
#define LOAD_KEY_VALUES(ValueType,DataType, key_value, dst) \
    dst = get_value<tff::core::model::ModelMetaKV,ValueType, DataType>(key_value, ctx)
#define LOAD_KEY_VALUE(ValueType,DataType, key_value, dst) \
    dst = LOAD_KEY_VALUES(ValueType,DataType, key_value, dst)[0]

    // model_list.h
#define FOR_EACH_MODEL(X) \
X(TFF_MODEL_ARCH_UNKNOWN, "unknown")\
X(TFF_MODEL_ARCH_LLAMA,   "LlamaForCausalLM") \
X(TFF_MODEL_ARCH_QWEN,    "QWenLMHeadModel") \
X(TFF_MODEL_ARCH_MISTRAL, "MistralForCausalLM") \
X(TFF_MODEL_ARCH_GEMMA,   "GemmaForCausalLM")

    enum class ModelArchitectureType {
#define DEFINE_ENUM(name, type_str) name,
        FOR_EACH_MODEL(DEFINE_ENUM)
#undef DEFINE_ENUM
        TFF_MODEL_ARCH_COUNT
    };

    enum ModelType {
        TFF_MODEL_TYPE_UNKNOWN = 0,
        TFF_MODEL_TYPE_8B = 1,
    };

    enum ModelAttentionSWAType {
        TFF_SWA_TYPE_NONE = 0,
        TFF_SWA_TYPE_STANDARD = 1,
        TFF_SWA_TYPE_CHUNKED = 2,
        TFF_SWA_TYPE_SYMMETRIC = 3,
    };

    struct ModelConfig {
        // 架构相关参数
        std::string _arch_name;
        std::vector<std::string> _architectures;
        //
        bool _vocab_only{};
        bool _rope_fine_tuned{};
        bool _use_mmap; // use mmap if possible
        bool _use_mlock; // force system to keep model in RAM
        bool _check_tensors; // validate model tensor data
        int32_t n_gpu_layers;


        uint32_t _n_ctx_train = 0;
        uint32_t _n_embd{};
        uint32_t _n_embd_features = 0;
        uint32_t _n_layer{};
        uint32_t _n_rot{};
        uint32_t _n_embd_head_k{};
        uint32_t _n_embd_head_v{};
        uint32_t _n_expert = 0;
        uint32_t _n_expert_used = 0;

        //
        std::vector<uint32_t> _n_head_arr;
        std::vector<uint32_t> _n_head_kv_arr;
        std::vector<uint32_t> _n_ff_arr;

        ModelAttentionSWAType _swa_type = TFF_SWA_TYPE_NONE;
        uint32_t _n_swa = 0;
        std::unordered_map<uint32_t, bool> _swa_layers;
        //
        tff::core::memory::DataType _kv_data_type;
    };

    enum TokenType {
        TFF_TOKEN_TYPE_UNDEFINED = 0,
        TFF_TOKEN_TYPE_NORMAL = 1,
        TFF_TOKEN_TYPE_UNKNOWN = 2,
        TFF_TOKEN_TYPE_CONTROL = 3,
        TFF_TOKEN_TYPE_USER_DEFINED = 4,
        TFF_TOKEN_TYPE_UNUSED = 5,
        TFF_TOKEN_TYPE_BYTE = 6,
    };

    enum TokenAttribute {
        TFF_TOKEN_ATTR_UNDEFINED = 1 << 0,
        TFF_TOKEN_ATTR_NORMAL = 1 << 1,
        TFF_TOKEN_ATTR_UNKNOWN = 1 << 2,
        TFF_TOKEN_ATTR_CONTROL = 1 << 3,
        TFF_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        TFF_TOKEN_ATTR_UNUSED = 1 << 5,
        TFF_TOKEN_ATTR_BYTE = 1 << 6,
        TFF_TOKEN_ATTR_NORMALIZED = 1 << 7,
        TFF_TOKEN_ATTR_LSTRIP = 1 << 8,
        TFF_TOKEN_ATTR_RSTRIP = 1 << 9,
        TFF_TOKEN_ATTR_SINGLE_WORD = 1 << 10,
    };

    // pre-tokenization types
    enum VocabPreType {
        TFF_VOCAB_PRE_TYPE_DEFAULT = 0,
        TFF_VOCAB_PRE_TYPE_LLAMA3 = 1,
        TFF_VOCAB_PRE_TYPE_DEEPSEEK_LLM = 2,
        TFF_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
        TFF_VOCAB_PRE_TYPE_FALCON = 4,
        TFF_VOCAB_PRE_TYPE_MPT = 5,
        TFF_VOCAB_PRE_TYPE_STARCODER = 6,
        TFF_VOCAB_PRE_TYPE_GPT2 = 7,
        TFF_VOCAB_PRE_TYPE_REFACT = 8,
        TFF_VOCAB_PRE_TYPE_COMMAND_R = 9,
        TFF_VOCAB_PRE_TYPE_STABLELM2 = 10,
        TFF_VOCAB_PRE_TYPE_QWEN2 = 11,
        TFF_VOCAB_PRE_TYPE_OLMO = 12,
        TFF_VOCAB_PRE_TYPE_DBRX = 13,
        TFF_VOCAB_PRE_TYPE_SMAUG = 14,
        TFF_VOCAB_PRE_TYPE_PORO = 15,
        TFF_VOCAB_PRE_TYPE_CHATGLM3 = 16,
        TFF_VOCAB_PRE_TYPE_CHATGLM4 = 17,
        TFF_VOCAB_PRE_TYPE_VIKING = 18,
        TFF_VOCAB_PRE_TYPE_JAIS = 19,
        TFF_VOCAB_PRE_TYPE_TEKKEN = 20,
        TFF_VOCAB_PRE_TYPE_SMOLLM = 21,
        TFF_VOCAB_PRE_TYPE_CODESHELL = 22,
        TFF_VOCAB_PRE_TYPE_BLOOM = 23,
        TFF_VOCAB_PRE_TYPE_GPT3_FINNISH = 24,
        TFF_VOCAB_PRE_TYPE_EXAONE = 25,
        TFF_VOCAB_PRE_TYPE_CHAMELEON = 26,
        TFF_VOCAB_PRE_TYPE_MINERVA = 27,
        TFF_VOCAB_PRE_TYPE_DEEPSEEK3_LLM = 28,
        TFF_VOCAB_PRE_TYPE_GPT4O = 29,
        TFF_VOCAB_PRE_TYPE_SUPERBPE = 30,
        TFF_VOCAB_PRE_TYPE_TRILLION = 31,
        TFF_VOCAB_PRE_TYPE_BAILINGMOE = 32,
        TFF_VOCAB_PRE_TYPE_LLAMA4 = 33,
        TFF_VOCAB_PRE_TYPE_PIXTRAL = 34,
        TFF_VOCAB_PRE_TYPE_SEED_CODER = 35,
        TFF_VOCAB_PRE_TYPE_HUNYUAN = 36,
        TFF_VOCAB_PRE_TYPE_KIMI_K2 = 37,
        TFF_VOCAB_PRE_TYPE_HUNYUAN_DENSE = 38,
        TFF_VOCAB_PRE_TYPE_GROK_2 = 39,
    };

    enum VocabType {
        TFF_VOCAB_TYPE_NONE = 0, // For models without vocab
        TFF_VOCAB_TYPE_SPM = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        TFF_VOCAB_TYPE_BPE = 2, // GPT-2 tokenizer based on byte-level BPE
        TFF_VOCAB_TYPE_WPM = 3, // BERT tokenizer based on WordPiece
        TFF_VOCAB_TYPE_UGM = 4, // T5 tokenizer based on Unigram
        TFF_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
        TFF_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
        FF_VOCAB_TYPE_COUNT // 用于数组大小
    };

    enum RopeType {
        TFF_ROPE_TYPE_NONE = -1,
        TFF_ROPE_TYPE_NORM = 0,
        TFF_ROPE_TYPE_NEOX = 1,
        TFF_ROPE_TYPE_MROPE = 2,
        TFF_ROPE_TYPE_VISION = 3,
    };

    struct TokenData {
        std::string _text;
        TokenAttribute _attribute;
        float _score;
    };

    //
    struct ModelWeight {
        uint16_t _idx;
        size_t _offs;
        std::shared_ptr<tff::core::memory::Tensor> _tensor_ptr;
    };

    //
    struct GGUFTensorInfo {
        std::string _name;
        uint64_t _offset;
        size_t _byte_size;
        std::shared_ptr<tff::core::memory::Tensor> _tensor_ptr;
    };

    //
    struct ModelContext {
        uint32_t _version = GGUF_VERSION;

        using BasicType = std::variant<
            uint8_t, int8_t, uint16_t, int16_t,
            uint32_t, int32_t, uint64_t, int64_t,
            float, double, bool, std::string
        >;

        using KVValue = std::variant<
            BasicType,
            std::vector<uint8_t>, std::vector<int8_t>,
            std::vector<uint16_t>, std::vector<int16_t>,
            std::vector<uint32_t>, std::vector<int32_t>,
            std::vector<uint64_t>, std::vector<int64_t>,
            std::vector<float>, std::vector<double>,
            std::vector<bool>, std::vector<std::string>
        >;

        std::unordered_map<std::string, KVValue> _kv;

        std::vector<GGUFTensorInfo> _tensor_info;

        size_t _alignment = GGUF_DEFAULT_ALIGNMENT;
        size_t _offset = 0;
        size_t _size = 0;
        size_t _max_tensor_bytesize = INT_MIN;

        std::shared_ptr<tff::core::memory::Memory> _data_memory_ptr;

        bool check_version() const {
            if ((_version & 0x0000FFFF) == 0x00000000) {
                tff::log::Logger::error(
                    "%s: failed to load model: this GGUF file version %d  is extremely large, is there a mismatch between the host and model endianness?\n",
                    __func__, _version);
                return false;
            }

            if (_version == 1) {
                tff::log::Logger::error("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n",
                                        __func__);
                return false;
            }
            if (_version > GGUF_VERSION) {
                tff::log::Logger::error(
                    "%s: this GGUF file is version %d but this software only supports up to version %d\n",
                    __func__, _version, GGUF_VERSION);
                return false;
            }

            return true;
        }
    };

    //
    enum ModelMetaKV {
        LLM_KV_GENERAL_TYPE,
        LLM_KV_GENERAL_ARCHITECTURE,
        LLM_KV_GENERAL_QUANTIZATION_VERSION,
        LLM_KV_GENERAL_ALIGNMENT,
        LLM_KV_GENERAL_FILE_TYPE,
        LLM_KV_GENERAL_NAME,
        LLM_KV_GENERAL_AUTHOR,
        LLM_KV_GENERAL_VERSION,
        LLM_KV_GENERAL_URL,
        LLM_KV_GENERAL_DESCRIPTION,
        LLM_KV_GENERAL_LICENSE,
        LLM_KV_GENERAL_SOURCE_URL,
        LLM_KV_GENERAL_SOURCE_HF_REPO,

        LLM_KV_VOCAB_SIZE,
        LLM_KV_CONTEXT_LENGTH,
        LLM_KV_EMBEDDING_LENGTH,
        LLM_KV_FEATURES_LENGTH,
        LLM_KV_BLOCK_COUNT,
        LLM_KV_LEADING_DENSE_BLOCK_COUNT,
        LLM_KV_FEED_FORWARD_LENGTH,
        LLM_KV_EXPERT_FEED_FORWARD_LENGTH,
        LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH,
        LLM_KV_USE_PARALLEL_RESIDUAL,
        LLM_KV_TENSOR_DATA_LAYOUT,
        LLM_KV_EXPERT_COUNT,
        LLM_KV_EXPERT_USED_COUNT,
        LLM_KV_EXPERT_SHARED_COUNT,
        LLM_KV_EXPERT_WEIGHTS_SCALE,
        LLM_KV_EXPERT_WEIGHTS_NORM,
        LLM_KV_EXPERT_GATING_FUNC,
        LLM_KV_MOE_EVERY_N_LAYERS,
        LLM_KV_NEXTN_PREDICT_LAYERS,
        LLM_KV_POOLING_TYPE,
        LLM_KV_LOGIT_SCALE,
        LLM_KV_DECODER_START_TOKEN_ID,
        LLM_KV_DECODER_BLOCK_COUNT,
        LLM_KV_ATTN_LOGIT_SOFTCAPPING,
        LLM_KV_ROUTER_LOGIT_SOFTCAPPING,
        LLM_KV_FINAL_LOGIT_SOFTCAPPING,
        LLM_KV_SWIN_NORM,
        LLM_KV_RESCALE_EVERY_N_LAYERS,
        LLM_KV_TIME_MIX_EXTRA_DIM,
        LLM_KV_TIME_DECAY_EXTRA_DIM,
        LLM_KV_RESIDUAL_SCALE,
        LLM_KV_EMBEDDING_SCALE,
        LLM_KV_TOKEN_SHIFT_COUNT,
        LLM_KV_INTERLEAVE_MOE_LAYER_STEP,

        LLM_KV_ATTENTION_HEAD_COUNT,
        LLM_KV_ATTENTION_HEAD_COUNT_KV,
        LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
        LLM_KV_ATTENTION_CLAMP_KQV,
        LLM_KV_ATTENTION_KEY_LENGTH,
        LLM_KV_ATTENTION_VALUE_LENGTH,
        LLM_KV_ATTENTION_LAYERNORM_EPS,
        LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,
        LLM_KV_ATTENTION_GROUPNORM_EPS,
        LLM_KV_ATTENTION_GROUPNORM_GROUPS,
        LLM_KV_ATTENTION_CAUSAL,
        LLM_KV_ATTENTION_Q_LORA_RANK,
        LLM_KV_ATTENTION_KV_LORA_RANK,
        LLM_KV_ATTENTION_DECAY_LORA_RANK,
        LLM_KV_ATTENTION_ICLR_LORA_RANK,
        LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK,
        LLM_KV_ATTENTION_GATE_LORA_RANK,
        LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT,
        LLM_KV_ATTENTION_SLIDING_WINDOW,
        LLM_KV_ATTENTION_SCALE,
        LLM_KV_ATTENTION_OUTPUT_SCALE,
        LLM_KV_ATTENTION_TEMPERATURE_LENGTH,
        LLM_KV_ATTENTION_KEY_LENGTH_MLA,
        LLM_KV_ATTENTION_VALUE_LENGTH_MLA,

        LLM_KV_ROPE_DIMENSION_COUNT,
        LLM_KV_ROPE_DIMENSION_SECTIONS,
        LLM_KV_ROPE_FREQ_BASE,
        LLM_KV_ROPE_SCALE_LINEAR,
        LLM_KV_ROPE_SCALING_TYPE,
        LLM_KV_ROPE_SCALING_FACTOR,
        LLM_KV_ROPE_SCALING_ATTN_FACTOR,
        LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,
        LLM_KV_ROPE_SCALING_FINETUNED,
        LLM_KV_ROPE_SCALING_YARN_LOG_MUL,
        LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,
        LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR,
        LLM_KV_ROPE_SCALING_YARN_BETA_FAST,
        LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,

        LLM_KV_SPLIT_NO,
        LLM_KV_SPLIT_COUNT,
        LLM_KV_SPLIT_TENSORS_COUNT,

        LLM_KV_SSM_INNER_SIZE,
        LLM_KV_SSM_CONV_KERNEL,
        LLM_KV_SSM_STATE_SIZE,
        LLM_KV_SSM_TIME_STEP_RANK,
        LLM_KV_SSM_GROUP_COUNT,
        LLM_KV_SSM_DT_B_C_RMS,

        LLM_KV_WKV_HEAD_SIZE,

        LLM_KV_TOKENIZER_MODEL,
        LLM_KV_TOKENIZER_PRE,
        LLM_KV_TOKENIZER_LIST,
        LLM_KV_TOKENIZER_TOKEN_TYPE,
        LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,
        LLM_KV_TOKENIZER_SCORES,
        LLM_KV_TOKENIZER_MERGES,
        LLM_KV_TOKENIZER_BOS_ID,
        LLM_KV_TOKENIZER_EOS_ID,
        LLM_KV_TOKENIZER_EOT_ID,
        LLM_KV_TOKENIZER_EOM_ID,
        LLM_KV_TOKENIZER_UNK_ID,
        LLM_KV_TOKENIZER_SEP_ID,
        LLM_KV_TOKENIZER_PAD_ID,
        LLM_KV_TOKENIZER_CLS_ID,
        LLM_KV_TOKENIZER_MASK_ID,
        LLM_KV_TOKENIZER_ADD_BOS,
        LLM_KV_TOKENIZER_ADD_EOS,
        LLM_KV_TOKENIZER_ADD_SEP,
        LLM_KV_TOKENIZER_ADD_PREFIX,
        LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,
        LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP,
        LLM_KV_TOKENIZER_HF_JSON,
        LLM_KV_TOKENIZER_RWKV,
        LLM_KV_TOKENIZER_CHAT_TEMPLATE,
        LLM_KV_TOKENIZER_FIM_PRE_ID,
        LLM_KV_TOKENIZER_FIM_SUF_ID,
        LLM_KV_TOKENIZER_FIM_MID_ID,
        LLM_KV_TOKENIZER_FIM_PAD_ID,
        LLM_KV_TOKENIZER_FIM_REP_ID,
        LLM_KV_TOKENIZER_FIM_SEP_ID,

        LLM_KV_ADAPTER_TYPE,
        LLM_KV_ADAPTER_LORA_ALPHA,
        LLM_KV_ADAPTER_LORA_TASK_NAME,
        LLM_KV_ADAPTER_LORA_PROMPT_PREFIX,
        LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS,

        LLM_KV_POSNET_EMBEDDING_LENGTH,
        LLM_KV_POSNET_BLOCK_COUNT,

        LLM_KV_CONVNEXT_EMBEDDING_LENGTH,
        LLM_KV_CONVNEXT_BLOCK_COUNT,

        LLM_KV_CLASSIFIER_OUTPUT_LABELS,

        LLM_KV_SHORTCONV_L_CACHE,
    };



    //
    enum ModelTensorLayerType {
        LLM_TENSOR_LAYER_NONE,
        LLM_TENSOR_LAYER_INPUT,
        LLM_TENSOR_LAYER_REPEATING,
        LLM_TENSOR_LAYER_OUTPUT,
    };

    //
}
#endif //TFFINFER_MODEL_BASEDEFINE_H
