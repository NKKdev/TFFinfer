//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_MODEL_BASEDEFINE_H
#define TFFINFER_MODEL_BASEDEFINE_H
#include <cstdint>
#include <string>
#include <variant>

#include "mem/Tensor.h"
#include "GGUFDef.h"
#include "Logger.h"
#include "LLAMADefine.h"
namespace tff::core::model {
#define TFF_TENSOR_MAX_DIMS 4

    enum ModelType {
        TFF_MODEL_TYPE_UNKNOWN = 0,
        TFF_MODEL_TYPE_8B = 1,
    };

    enum ModelArchitecture {
        TFF_MODEL_ARCH_UNKNOWN = 0,
        TFF_MODEL_ARCH_LLAMA = 1,
    };

    struct ModelHeadParams {
        bool _vocab_only{};
        bool _rope_fine_tuned{};


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
    };

    struct ModelParams {
        bool _vocab_only; // only load the vocabulary, no weights
        bool _use_mmap; // use mmap if possible
        bool _use_mlock; // force system to keep model in RAM
        bool _check_tensors; // validate model tensor data
        int32_t n_gpu_layers;
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
        TFF_TOKEN_ATTR_UNDEFINED = 0,
        TFF_TOKEN_ATTR_UNKNOWN = 1 << 0,
        TFF_TOKEN_ATTR_UNUSED = 1 << 1,
        TFF_TOKEN_ATTR_NORMAL = 1 << 2,
        TFF_TOKEN_ATTR_CONTROL = 1 << 3,
        TFF_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        TFF_TOKEN_ATTR_BYTE = 1 << 5,
        TFF_TOKEN_ATTR_NORMALIZED = 1 << 6,
        TFF_TOKEN_ATTR_LSTRIP = 1 << 7,
        TFF_TOKEN_ATTR_RSTRIP = 1 << 8,
        TFF_TOKEN_ATTR_SINGLE_WORD = 1 << 9,
    };

    struct TokenData {
        std::string _text;
        TokenAttribute _attribute;
        float _score;
    };

    //
    struct ModelWeights {
        uint16_t _idx;
        size_t _offs;
        std::shared_ptr<tff::core::memory::Tensor> _tensor_ptr;
    };

    //
    struct GGUFTensorInfo {
        std::string _name;
        uint64_t _offset;
        std::shared_ptr<tff::core::memory::Tensor> _tensor_ptr;
    };

    //
    struct GGUFContext {
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

        void *_data = nullptr;

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

    enum ModelTensorType {
        LLM_TENSOR_TOKEN_EMBD,
        LLM_TENSOR_TOKEN_EMBD_NORM,
        LLM_TENSOR_TOKEN_TYPES,
        LLM_TENSOR_POS_EMBD,
        LLM_TENSOR_OUTPUT,
        LLM_TENSOR_OUTPUT_NORM,
        LLM_TENSOR_ROPE_FREQS,
        LLM_TENSOR_ROPE_FACTORS_LONG,
        LLM_TENSOR_ROPE_FACTORS_SHORT,
        LLM_TENSOR_ATTN_Q,
        LLM_TENSOR_ATTN_K,
        LLM_TENSOR_ATTN_V,
        LLM_TENSOR_ATTN_QKV,
        LLM_TENSOR_ATTN_OUT,
        LLM_TENSOR_ATTN_NORM,
        LLM_TENSOR_ATTN_NORM_2,
        LLM_TENSOR_ATTN_OUT_NORM,
        LLM_TENSOR_ATTN_POST_NORM,
        LLM_TENSOR_ATTN_ROT_EMBD,
        LLM_TENSOR_ATTN_SINKS,
        LLM_TENSOR_FFN_GATE_INP,
        LLM_TENSOR_FFN_GATE_INP_SHEXP,
        LLM_TENSOR_FFN_NORM,
        LLM_TENSOR_FFN_POST_NORM,
        LLM_TENSOR_FFN_GATE,
        LLM_TENSOR_FFN_DOWN,
        LLM_TENSOR_FFN_UP,
        LLM_TENSOR_FFN_ACT,
        LLM_TENSOR_FFN_DOWN_EXP, // split experts for backward compatibility
        LLM_TENSOR_FFN_GATE_EXP,
        LLM_TENSOR_FFN_UP_EXP,
        LLM_TENSOR_FFN_NORM_EXPS,
        LLM_TENSOR_FFN_DOWN_EXPS, // merged experts
        LLM_TENSOR_FFN_GATE_EXPS,
        LLM_TENSOR_FFN_UP_EXPS,
        LLM_TENSOR_FFN_DOWN_SHEXP,
        LLM_TENSOR_FFN_GATE_SHEXP,
        LLM_TENSOR_FFN_UP_SHEXP,
        LLM_TENSOR_FFN_EXP_PROBS_B,
        LLM_TENSOR_ATTN_Q_NORM,
        LLM_TENSOR_ATTN_K_NORM,
        LLM_TENSOR_LAYER_OUT_NORM,
        LLM_TENSOR_POST_ATTN_NORM,
        LLM_TENSOR_POST_MLP_NORM,
        LLM_TENSOR_PER_LAYER_TOKEN_EMBD, // gemma3n
        LLM_TENSOR_PER_LAYER_MODEL_PROJ, // gemma3n
        LLM_TENSOR_PER_LAYER_INP_GATE, // gemma3n
        LLM_TENSOR_PER_LAYER_PROJ, // gemma3n
        LLM_TENSOR_PER_LAYER_PROJ_NORM, // gemma3n
        LLM_TENSOR_PER_LAYER_POST_NORM, // gemma3n
        LLM_TENSOR_ALTUP_PROJ, // gemma3n
        LLM_TENSOR_ALTUP_UNEMBD_PROJ, // gemma3n
        LLM_TENSOR_ALTUP_CORRECT_COEF, // gemma3n
        LLM_TENSOR_ALTUP_CORRECT_SCALE, // gemma3n
        LLM_TENSOR_ALTUP_PREDICT_COEF, // gemma3n
        LLM_TENSOR_ALTUP_ROUTER, // gemma3n
        LLM_TENSOR_ALTUP_ROUTER_NORM, // gemma3n
        LLM_TENSOR_LAUREL_L, // gemma3n
        LLM_TENSOR_LAUREL_R, // gemma3n
        LLM_TENSOR_LAUREL_POST_NORM, // gemma3n
        LLM_TENSOR_SSM_IN,
        LLM_TENSOR_SSM_CONV1D,
        LLM_TENSOR_SSM_X,
        LLM_TENSOR_SSM_DT,
        LLM_TENSOR_SSM_DT_NORM,
        LLM_TENSOR_SSM_A,
        LLM_TENSOR_SSM_B_NORM,
        LLM_TENSOR_SSM_C_NORM,
        LLM_TENSOR_SSM_D,
        LLM_TENSOR_SSM_NORM,
        LLM_TENSOR_SSM_OUT,
        LLM_TENSOR_TIME_MIX_W0,
        LLM_TENSOR_TIME_MIX_W1,
        LLM_TENSOR_TIME_MIX_W2,
        LLM_TENSOR_TIME_MIX_A0,
        LLM_TENSOR_TIME_MIX_A1,
        LLM_TENSOR_TIME_MIX_A2,
        LLM_TENSOR_TIME_MIX_V0,
        LLM_TENSOR_TIME_MIX_V1,
        LLM_TENSOR_TIME_MIX_V2,
        LLM_TENSOR_TIME_MIX_G1,
        LLM_TENSOR_TIME_MIX_G2,
        LLM_TENSOR_TIME_MIX_K_K,
        LLM_TENSOR_TIME_MIX_K_A,
        LLM_TENSOR_TIME_MIX_R_K,
        LLM_TENSOR_TIME_MIX_LERP_X,
        LLM_TENSOR_TIME_MIX_LERP_W,
        LLM_TENSOR_TIME_MIX_LERP_K,
        LLM_TENSOR_TIME_MIX_LERP_V,
        LLM_TENSOR_TIME_MIX_LERP_R,
        LLM_TENSOR_TIME_MIX_LERP_G,
        LLM_TENSOR_TIME_MIX_LERP_FUSED,
        LLM_TENSOR_TIME_MIX_FIRST,
        LLM_TENSOR_TIME_MIX_DECAY,
        LLM_TENSOR_TIME_MIX_DECAY_W1,
        LLM_TENSOR_TIME_MIX_DECAY_W2,
        LLM_TENSOR_TIME_MIX_KEY,
        LLM_TENSOR_TIME_MIX_VALUE,
        LLM_TENSOR_TIME_MIX_RECEPTANCE,
        LLM_TENSOR_TIME_MIX_GATE,
        LLM_TENSOR_TIME_MIX_LN,
        LLM_TENSOR_TIME_MIX_OUTPUT,
        LLM_TENSOR_CHANNEL_MIX_LERP_K,
        LLM_TENSOR_CHANNEL_MIX_LERP_R,
        LLM_TENSOR_CHANNEL_MIX_KEY,
        LLM_TENSOR_CHANNEL_MIX_RECEPTANCE,
        LLM_TENSOR_CHANNEL_MIX_VALUE,
        LLM_TENSOR_ATTN_Q_A,
        LLM_TENSOR_ATTN_Q_B,
        LLM_TENSOR_ATTN_KV_A_MQA,
        LLM_TENSOR_ATTN_KV_B,
        LLM_TENSOR_ATTN_K_B,
        LLM_TENSOR_ATTN_V_B,
        LLM_TENSOR_ATTN_Q_A_NORM,
        LLM_TENSOR_ATTN_KV_A_NORM,
        LLM_TENSOR_ATTN_SUB_NORM,
        LLM_TENSOR_FFN_SUB_NORM,
        LLM_TENSOR_DEC_ATTN_NORM,
        LLM_TENSOR_DEC_ATTN_Q,
        LLM_TENSOR_DEC_ATTN_K,
        LLM_TENSOR_DEC_ATTN_V,
        LLM_TENSOR_DEC_ATTN_OUT,
        LLM_TENSOR_DEC_ATTN_REL_B,
        LLM_TENSOR_DEC_CROSS_ATTN_NORM,
        LLM_TENSOR_DEC_CROSS_ATTN_Q,
        LLM_TENSOR_DEC_CROSS_ATTN_K,
        LLM_TENSOR_DEC_CROSS_ATTN_V,
        LLM_TENSOR_DEC_CROSS_ATTN_OUT,
        LLM_TENSOR_DEC_CROSS_ATTN_REL_B,
        LLM_TENSOR_DEC_FFN_NORM,
        LLM_TENSOR_DEC_FFN_GATE,
        LLM_TENSOR_DEC_FFN_DOWN,
        LLM_TENSOR_DEC_FFN_UP,
        LLM_TENSOR_DEC_OUTPUT_NORM,
        LLM_TENSOR_ENC_ATTN_NORM,
        LLM_TENSOR_ENC_ATTN_Q,
        LLM_TENSOR_ENC_ATTN_K,
        LLM_TENSOR_ENC_ATTN_V,
        LLM_TENSOR_ENC_ATTN_OUT,
        LLM_TENSOR_ENC_ATTN_REL_B,
        LLM_TENSOR_ENC_FFN_NORM,
        LLM_TENSOR_ENC_FFN_GATE,
        LLM_TENSOR_ENC_FFN_DOWN,
        LLM_TENSOR_ENC_FFN_UP,
        LLM_TENSOR_ENC_OUTPUT_NORM,
        LLM_TENSOR_CLS,
        LLM_TENSOR_CLS_OUT,
        LLM_TENSOR_CONV1D,
        LLM_TENSOR_CONVNEXT_DW,
        LLM_TENSOR_CONVNEXT_NORM,
        LLM_TENSOR_CONVNEXT_PW1,
        LLM_TENSOR_CONVNEXT_PW2,
        LLM_TENSOR_CONVNEXT_GAMMA,
        LLM_TENSOR_POS_NET_CONV1,
        LLM_TENSOR_POS_NET_CONV2,
        LLM_TENSOR_POS_NET_NORM,
        LLM_TENSOR_POS_NET_NORM1,
        LLM_TENSOR_POS_NET_NORM2,
        LLM_TENSOR_POS_NET_ATTN_NORM,
        LLM_TENSOR_POS_NET_ATTN_Q,
        LLM_TENSOR_POS_NET_ATTN_K,
        LLM_TENSOR_POS_NET_ATTN_V,
        LLM_TENSOR_POS_NET_ATTN_OUT,
        LLM_TENSOR_SHORTCONV_CONV,
        LLM_TENSOR_SHORTCONV_INPROJ,
        LLM_TENSOR_SHORTCONV_OUTPROJ,
        LLM_TENSOR_NEXTN_EH_PROJ,
        LLM_TENSOR_NEXTN_EMBED_TOKENS,
        LLM_TENSOR_NEXTN_ENORM,
        LLM_TENSOR_NEXTN_HNORM,
        LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,
        LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,
    };

    //
    enum ModelTensorLayerType {
        LLM_TENSOR_LAYER_INPUT,
        LLM_TENSOR_LAYER_REPEATING,
        LLM_TENSOR_LAYER_OUTPUT,
    };
    //

}
#endif //TFFINFER_MODEL_BASEDEFINE_H
