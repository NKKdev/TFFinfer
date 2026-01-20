//
// Created by nkk on 2025/10/24.
//

#ifndef TFFINFER_MODELGLOBALVAR_H
#define TFFINFER_MODELGLOBALVAR_H
#include <unordered_map>
#include "../model/BaseDefine.h"
#include "graph/GraphNode.h"
#include "graph/BaseDefine.h"
#include "mem/BaseDefine.h"
#include "../model/LLAMADefine.h"
#include "fmt/format.h"
using namespace tff::core::model;

namespace tff::core::global {
    enum TaskFlowType {
        TFF_FLOW_LLM,
        TFF_FLOW_CNN,
        TFF_FLOW_COUNT
    };

    static const std::unordered_map<tff::core::model::ModelArchitectureType, const char *> LLM_ARCH_NAMES = {
        {tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_UNKNOWN, "unknow"},
        {tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA, "llama"},
        {tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_QWEN3, "qwen3"},
    };
    static const std::unordered_map<tff::core::model::ModelMetaKV, const char *> LLM_KV_NAMES = {
        {LLM_KV_GENERAL_TYPE, "general.type"},
        {LLM_KV_GENERAL_ARCHITECTURE, "general.architecture"},
        {LLM_KV_GENERAL_QUANTIZATION_VERSION, "general.quantization_version"},
        {LLM_KV_GENERAL_ALIGNMENT, "general.alignment"},
        {LLM_KV_GENERAL_FILE_TYPE, "general.file_type"},
        {LLM_KV_GENERAL_NAME, "general.name"},
        {LLM_KV_GENERAL_AUTHOR, "general.author"},
        {LLM_KV_GENERAL_VERSION, "general.version"},
        {LLM_KV_GENERAL_URL, "general.url"},
        {LLM_KV_GENERAL_DESCRIPTION, "general.description"},
        {LLM_KV_GENERAL_LICENSE, "general.license"},
        {LLM_KV_GENERAL_SOURCE_URL, "general.source.url"},
        {LLM_KV_GENERAL_SOURCE_HF_REPO, "general.source.huggingface.repository"},

        {LLM_KV_VOCAB_SIZE, "%s.vocab_size"},
        {LLM_KV_CONTEXT_LENGTH, "%s.context_length"},
        {LLM_KV_EMBEDDING_LENGTH, "%s.embedding_length"},
        {LLM_KV_FEATURES_LENGTH, "%s.features_length"},
        {LLM_KV_BLOCK_COUNT, "%s.block_count"},
        {LLM_KV_LEADING_DENSE_BLOCK_COUNT, "%s.leading_dense_block_count"},
        {LLM_KV_FEED_FORWARD_LENGTH, "%s.feed_forward_length"},
        {LLM_KV_EXPERT_FEED_FORWARD_LENGTH, "%s.expert_feed_forward_length"},
        {LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, "%s.expert_shared_feed_forward_length"},
        {LLM_KV_USE_PARALLEL_RESIDUAL, "%s.use_parallel_residual"},
        {LLM_KV_TENSOR_DATA_LAYOUT, "%s.tensor_data_layout"},
        {LLM_KV_EXPERT_COUNT, "%s.expert_count"},
        {LLM_KV_EXPERT_USED_COUNT, "%s.expert_used_count"},
        {LLM_KV_EXPERT_SHARED_COUNT, "%s.expert_shared_count"},
        {LLM_KV_EXPERT_WEIGHTS_SCALE, "%s.expert_weights_scale"},
        {LLM_KV_EXPERT_WEIGHTS_NORM, "%s.expert_weights_norm"},
        {LLM_KV_EXPERT_GATING_FUNC, "%s.expert_gating_func"},
        {LLM_KV_MOE_EVERY_N_LAYERS, "%s.moe_every_n_layers"},
        {LLM_KV_NEXTN_PREDICT_LAYERS, "%s.nextn_predict_layers"},
        {LLM_KV_POOLING_TYPE, "%s.pooling_type"},
        {LLM_KV_LOGIT_SCALE, "%s.logit_scale"},
        {LLM_KV_DECODER_START_TOKEN_ID, "%s.decoder_start_token_id"},
        {LLM_KV_DECODER_BLOCK_COUNT, "%s.decoder_block_count"},
        {LLM_KV_ATTN_LOGIT_SOFTCAPPING, "%s.attn_logit_softcapping"},
        {LLM_KV_ROUTER_LOGIT_SOFTCAPPING, "%s.router_logit_softcapping"},
        {LLM_KV_FINAL_LOGIT_SOFTCAPPING, "%s.final_logit_softcapping"},
        {LLM_KV_SWIN_NORM, "%s.swin_norm"},
        {LLM_KV_RESCALE_EVERY_N_LAYERS, "%s.rescale_every_n_layers"},
        {LLM_KV_TIME_MIX_EXTRA_DIM, "%s.time_mix_extra_dim"},
        {LLM_KV_TIME_DECAY_EXTRA_DIM, "%s.time_decay_extra_dim"},
        {LLM_KV_RESIDUAL_SCALE, "%s.residual_scale"},
        {LLM_KV_EMBEDDING_SCALE, "%s.embedding_scale"},
        {LLM_KV_TOKEN_SHIFT_COUNT, "%s.token_shift_count"},
        {LLM_KV_INTERLEAVE_MOE_LAYER_STEP, "%s.interleave_moe_layer_step"},

        {LLM_KV_ATTENTION_HEAD_COUNT, "%s.attention.head_count"},
        {LLM_KV_ATTENTION_HEAD_COUNT_KV, "%s.attention.head_count_kv"},
        {LLM_KV_ATTENTION_MAX_ALIBI_BIAS, "%s.attention.max_alibi_bias"},
        {LLM_KV_ATTENTION_CLAMP_KQV, "%s.attention.clamp_kqv"},
        {LLM_KV_ATTENTION_KEY_LENGTH, "%s.attention.key_length"},
        {LLM_KV_ATTENTION_VALUE_LENGTH, "%s.attention.value_length"},
        {LLM_KV_ATTENTION_LAYERNORM_EPS, "%s.attention.layer_norm_epsilon"},
        {LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, "%s.attention.layer_norm_rms_epsilon"},
        {LLM_KV_ATTENTION_GROUPNORM_EPS, "%s.attention.group_norm_epsilon"},
        {LLM_KV_ATTENTION_GROUPNORM_GROUPS, "%s.attention.group_norm_groups"},
        {LLM_KV_ATTENTION_CAUSAL, "%s.attention.causal"},
        {LLM_KV_ATTENTION_Q_LORA_RANK, "%s.attention.q_lora_rank"},
        {LLM_KV_ATTENTION_KV_LORA_RANK, "%s.attention.kv_lora_rank"},
        {LLM_KV_ATTENTION_DECAY_LORA_RANK, "%s.attention.decay_lora_rank"},
        {LLM_KV_ATTENTION_ICLR_LORA_RANK, "%s.attention.iclr_lora_rank"},
        {LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, "%s.attention.value_residual_mix_lora_rank"},
        {LLM_KV_ATTENTION_GATE_LORA_RANK, "%s.attention.gate_lora_rank"},
        {LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, "%s.attention.relative_buckets_count"},
        {LLM_KV_ATTENTION_SLIDING_WINDOW, "%s.attention.sliding_window"},
        {LLM_KV_ATTENTION_SCALE, "%s.attention.scale"},
        {LLM_KV_ATTENTION_OUTPUT_SCALE, "%s.attention.output_scale"},
        {LLM_KV_ATTENTION_TEMPERATURE_LENGTH, "%s.attention.temperature_length"},
        {LLM_KV_ATTENTION_KEY_LENGTH_MLA, "%s.attention.key_length_mla"},
        {LLM_KV_ATTENTION_VALUE_LENGTH_MLA, "%s.attention.value_length_mla"},

        {LLM_KV_ROPE_DIMENSION_COUNT, "%s.rope.dimension_count"},
        {LLM_KV_ROPE_DIMENSION_SECTIONS, "%s.rope.dimension_sections"},
        {LLM_KV_ROPE_FREQ_BASE, "%s.rope.freq_base"},
        {LLM_KV_ROPE_SCALE_LINEAR, "%s.rope.scale_linear"},
        {LLM_KV_ROPE_SCALING_TYPE, "%s.rope.scaling.type"},
        {LLM_KV_ROPE_SCALING_FACTOR, "%s.rope.scaling.factor"},
        {LLM_KV_ROPE_SCALING_ATTN_FACTOR, "%s.rope.scaling.attn_factor"},
        {LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, "%s.rope.scaling.original_context_length"},
        {LLM_KV_ROPE_SCALING_FINETUNED, "%s.rope.scaling.finetuned"},
        {LLM_KV_ROPE_SCALING_YARN_LOG_MUL, "%s.rope.scaling.yarn_log_multiplier"},
        {LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR, "%s.rope.scaling.yarn_ext_factor"},
        {LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, "%s.rope.scaling.yarn_attn_factor"},
        {LLM_KV_ROPE_SCALING_YARN_BETA_FAST, "%s.rope.scaling.yarn_beta_fast"},
        {LLM_KV_ROPE_SCALING_YARN_BETA_SLOW, "%s.rope.scaling.yarn_beta_slow"},

        {LLM_KV_SPLIT_NO, "split.no"},
        {LLM_KV_SPLIT_COUNT, "split.count"},
        {LLM_KV_SPLIT_TENSORS_COUNT, "split.tensors.count"},

        {LLM_KV_SSM_CONV_KERNEL, "%s.ssm.conv_kernel"},
        {LLM_KV_SSM_INNER_SIZE, "%s.ssm.inner_size"},
        {LLM_KV_SSM_STATE_SIZE, "%s.ssm.state_size"},
        {LLM_KV_SSM_TIME_STEP_RANK, "%s.ssm.time_step_rank"},
        {LLM_KV_SSM_GROUP_COUNT, "%s.ssm.group_count"},
        {LLM_KV_SSM_DT_B_C_RMS, "%s.ssm.dt_b_c_rms"},

        {LLM_KV_WKV_HEAD_SIZE, "%s.wkv.head_size"},

        {LLM_KV_POSNET_EMBEDDING_LENGTH, "%s.posnet.embedding_length"},
        {LLM_KV_POSNET_BLOCK_COUNT, "%s.posnet.block_count"},

        {LLM_KV_CONVNEXT_EMBEDDING_LENGTH, "%s.convnext.embedding_length"},
        {LLM_KV_CONVNEXT_BLOCK_COUNT, "%s.convnext.block_count"},

        {LLM_KV_CLASSIFIER_OUTPUT_LABELS, "%s.classifier.output_labels"},

        {LLM_KV_SHORTCONV_L_CACHE, "%s.shortconv.l_cache"},

        {LLM_KV_TOKENIZER_MODEL, "tokenizer.ggml.model"},
        {LLM_KV_TOKENIZER_PRE, "tokenizer.ggml.pre"},
        {LLM_KV_TOKENIZER_LIST, "tokenizer.ggml.tokens"},
        {LLM_KV_TOKENIZER_TOKEN_TYPE, "tokenizer.ggml.token_type"},
        {LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, "tokenizer.ggml.token_type_count"},
        {LLM_KV_TOKENIZER_SCORES, "tokenizer.ggml.scores"},
        {LLM_KV_TOKENIZER_MERGES, "tokenizer.ggml.merges"},
        {LLM_KV_TOKENIZER_BOS_ID, "tokenizer.ggml.bos_token_id"},
        {LLM_KV_TOKENIZER_EOS_ID, "tokenizer.ggml.eos_token_id"},
        {LLM_KV_TOKENIZER_EOT_ID, "tokenizer.ggml.eot_token_id"},
        {LLM_KV_TOKENIZER_EOM_ID, "tokenizer.ggml.eom_token_id"},
        {LLM_KV_TOKENIZER_UNK_ID, "tokenizer.ggml.unknown_token_id"},
        {LLM_KV_TOKENIZER_SEP_ID, "tokenizer.ggml.seperator_token_id"},
        {LLM_KV_TOKENIZER_PAD_ID, "tokenizer.ggml.padding_token_id"},
        {LLM_KV_TOKENIZER_CLS_ID, "tokenizer.ggml.cls_token_id"},
        {LLM_KV_TOKENIZER_MASK_ID, "tokenizer.ggml.mask_token_id"},
        {LLM_KV_TOKENIZER_ADD_BOS, "tokenizer.ggml.add_bos_token"},
        {LLM_KV_TOKENIZER_ADD_EOS, "tokenizer.ggml.add_eos_token"},
        {LLM_KV_TOKENIZER_ADD_SEP, "tokenizer.ggml.add_sep_token"},
        {LLM_KV_TOKENIZER_ADD_PREFIX, "tokenizer.ggml.add_space_prefix"},
        {LLM_KV_TOKENIZER_REMOVE_EXTRA_WS, "tokenizer.ggml.remove_extra_whitespaces"},
        {LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP, "tokenizer.ggml.precompiled_charsmap"},
        {LLM_KV_TOKENIZER_HF_JSON, "tokenizer.huggingface.json"},
        {LLM_KV_TOKENIZER_RWKV, "tokenizer.rwkv.world"},
        {LLM_KV_TOKENIZER_CHAT_TEMPLATE, "tokenizer.chat_template"},
        {LLM_KV_TOKENIZER_FIM_PRE_ID, "tokenizer.ggml.fim_pre_token_id"},
        {LLM_KV_TOKENIZER_FIM_SUF_ID, "tokenizer.ggml.fim_suf_token_id"},
        {LLM_KV_TOKENIZER_FIM_MID_ID, "tokenizer.ggml.fim_mid_token_id"},
        {LLM_KV_TOKENIZER_FIM_PAD_ID, "tokenizer.ggml.fim_pad_token_id"},
        {LLM_KV_TOKENIZER_FIM_REP_ID, "tokenizer.ggml.fim_rep_token_id"},
        {LLM_KV_TOKENIZER_FIM_SEP_ID, "tokenizer.ggml.fim_sep_token_id"},

        {LLM_KV_ADAPTER_TYPE, "adapter.type"},
        {LLM_KV_ADAPTER_LORA_ALPHA, "adapter.lora.alpha"},
        {LLM_KV_ADAPTER_LORA_TASK_NAME, "adapter.lora.task_name"},
        {LLM_KV_ADAPTER_LORA_PROMPT_PREFIX, "adapter.lora.prompt_prefix"},
        {LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS, "adapter.alora.invocation_tokens"},
    };
    static const std::unordered_map<tff::core::model::ModelArchitectureType, std::unordered_map<
        tff::core::memory::ModelTensorType,
        const char *> > LLM_TENSOR_NAMES = {
        {
            tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA,
            {
                {tff::core::memory::LLM_TENSOR_TOKEN_EMBD, "token_embd"},
                {tff::core::memory::LLM_TENSOR_OUTPUT_NORM, "output_norm"},
                {tff::core::memory::LLM_TENSOR_OUTPUT, "output"},
                {tff::core::memory::LLM_TENSOR_ROPE_FREQS, "rope_freqs"},
                {tff::core::memory::LLM_TENSOR_ATTN_NORM, "blk.%d.attn_norm"},
                {tff::core::memory::LLM_TENSOR_ATTN_Q, "blk.%d.attn_q"},
                {tff::core::memory::LLM_TENSOR_ATTN_K, "blk.%d.attn_k"},
                {tff::core::memory::LLM_TENSOR_ATTN_V, "blk.%d.attn_v"},
                {tff::core::memory::LLM_TENSOR_ATTN_OUT, "blk.%d.attn_output"},
                {tff::core::memory::LLM_TENSOR_ATTN_ROT_EMBD, "blk.%d.attn_rot_embd"},
                {tff::core::memory::LLM_TENSOR_FFN_GATE_INP, "blk.%d.ffn_gate_inp"},
                {tff::core::memory::LLM_TENSOR_FFN_NORM, "blk.%d.ffn_norm"},
                {tff::core::memory::LLM_TENSOR_FFN_GATE, "blk.%d.ffn_gate"},
                {tff::core::memory::LLM_TENSOR_FFN_DOWN, "blk.%d.ffn_down"},
                {tff::core::memory::LLM_TENSOR_FFN_UP, "blk.%d.ffn_up"},
                {tff::core::memory::LLM_TENSOR_FFN_GATE_EXP, "blk.%d.ffn_gate.%d"},
                {tff::core::memory::LLM_TENSOR_FFN_DOWN_EXP, "blk.%d.ffn_down.%d"},
                {tff::core::memory::LLM_TENSOR_FFN_UP_EXP, "blk.%d.ffn_up.%d"},
                {tff::core::memory::LLM_TENSOR_FFN_GATE_EXPS, "blk.%d.ffn_gate_exps"},
                {tff::core::memory::LLM_TENSOR_FFN_DOWN_EXPS, "blk.%d.ffn_down_exps"},
                {tff::core::memory::LLM_TENSOR_FFN_UP_EXPS, "blk.%d.ffn_up_exps"},
            },
        },
        {
            tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_QWEN3,
            {
                {tff::core::memory::LLM_TENSOR_TOKEN_EMBD, "token_embd"},
                {tff::core::memory::LLM_TENSOR_OUTPUT_NORM, "output_norm"},
                {tff::core::memory::LLM_TENSOR_OUTPUT, "output"},
                {tff::core::memory::LLM_TENSOR_ROPE_FREQS, "rope_freqs"},
                {tff::core::memory::LLM_TENSOR_ATTN_NORM, "blk.%d.attn_norm"},
                {tff::core::memory::LLM_TENSOR_ATTN_Q, "blk.%d.attn_q"},
                {tff::core::memory::LLM_TENSOR_ATTN_K, "blk.%d.attn_k"},
                {tff::core::memory::LLM_TENSOR_ATTN_V, "blk.%d.attn_v"},
                {tff::core::memory::LLM_TENSOR_ATTN_Q_NORM, "blk.%d.attn_q_norm"},
                {tff::core::memory::LLM_TENSOR_ATTN_K_NORM, "blk.%d.attn_k_norm"},
                {tff::core::memory::LLM_TENSOR_ATTN_OUT, "blk.%d.attn_output"},
                {tff::core::memory::LLM_TENSOR_ATTN_ROT_EMBD, "blk.%d.attn_rot_embd"},
                {tff::core::memory::LLM_TENSOR_FFN_GATE_INP, "blk.%d.ffn_gate_inp"},
                {tff::core::memory::LLM_TENSOR_FFN_NORM, "blk.%d.ffn_norm"},
                {tff::core::memory::LLM_TENSOR_FFN_GATE, "blk.%d.ffn_gate"},
                {tff::core::memory::LLM_TENSOR_FFN_DOWN, "blk.%d.ffn_down"},
                {tff::core::memory::LLM_TENSOR_FFN_UP, "blk.%d.ffn_up"},
                {tff::core::memory::LLM_TENSOR_FFN_GATE_EXP, "blk.%d.ffn_gate.%d"},
                {tff::core::memory::LLM_TENSOR_FFN_DOWN_EXP, "blk.%d.ffn_down.%d"},
                {tff::core::memory::LLM_TENSOR_FFN_UP_EXP, "blk.%d.ffn_up.%d"},
                {tff::core::memory::LLM_TENSOR_FFN_GATE_EXPS, "blk.%d.ffn_gate_exps"},
                {tff::core::memory::LLM_TENSOR_FFN_DOWN_EXPS, "blk.%d.ffn_down_exps"},
                {tff::core::memory::LLM_TENSOR_FFN_UP_EXPS, "blk.%d.ffn_up_exps"},
            },
        },
    };
    static const std::unordered_map<tff::core::memory::ModelTensorType, std::pair<
        tff::core::model::ModelTensorLayerType, tff::core::graph::TffOpType> > LLM_LAYER_OP_INFOS = {
        {
            tff::core::memory::ModelTensorType::LLM_TENSOR_TOKEN_POS,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_MEM_REF}
        },
        {
            tff::core::memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_MEM_REF}
        },
        {
            tff::core::memory::LLM_TENSOR_TOKEN_EMBD,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_MEM_REF}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_EMBD,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_TOKEN_EMBD_NORM,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_TOKEN_TYPES,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {tff::core::memory::LLM_TENSOR_OUTPUT, {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}},
        {tff::core::memory::LLM_TENSOR_CLS, {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}},
        {tff::core::memory::LLM_TENSOR_CLS_OUT, {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}},
        {
            tff::core::memory::LLM_TENSOR_OUTPUT_NORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_RMS_NORM}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_QKV,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_DOWN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_UP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_DOWN_SHEXP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE_SHEXP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_UP_SHEXP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q_A,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_KV_A_MQA,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_KV_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_K_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_V_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_SINKS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_SCALE}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_FFN_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_FFN_DOWN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_FFN_UP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_FFN_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_FFN_DOWN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_FFN_UP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE_INP_SHEXP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE_INP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_IN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_X,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_DT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_W1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_W2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_A1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_A2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_V1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_V2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_G1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_G2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_DECAY_W1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_DECAY_W2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_KEY,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_VALUE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_RECEPTANCE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_OUTPUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_KEY,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_RECEPTANCE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_VALUE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {tff::core::memory::LLM_TENSOR_FFN_ACT, {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_DIV}},
        {
            tff::core::memory::LLM_TENSOR_SSM_CONV1D,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_SSM_CONV}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_A,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_SSM_SCAN}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_DT_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_B_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_C_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_D,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_X,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_LERP_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_LERP_R,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_K_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_K_A,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_R_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_W,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_R,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_G,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_FUSED,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_DECAY,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_W0,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_A0,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_V0,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_FIRST,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_RWKV_WKV6}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_NORM_2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_OUT_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_POST_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_POST_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_NORM_EXPS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_K_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_LAYER_OUT_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q_A_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_KV_A_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_SUB_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_SUB_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_FFN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_FFN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_REL_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_REL_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_DOWN_EXPS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT_ID}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE_EXPS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT_ID}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_UP_EXPS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT_ID}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_EXP_PROBS_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        // altup / laurel (gemma 3n)
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_TOKEN_EMBD,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_MODEL_PROJ,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_PROJ_NORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_PROJ,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_UNEMBD_PROJ,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_INP_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_PROJ,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_POST_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_CORRECT_COEF,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_CORRECT_SCALE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_PREDICT_COEF,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_ROUTER,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_ROUTER_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_LAUREL_L,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_LAUREL_R,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_LAUREL_POST_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        // this tensor is loaded for T5, but never used
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_REL_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_NONE}
        },
        {tff::core::memory::LLM_TENSOR_CONV1D, {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_IM2COL}},
        {
            tff::core::memory::LLM_TENSOR_POS_NET_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_NORM1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_NORM2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_CONV1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_IM2COL}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_CONV2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_IM2COL}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_DW,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_IM2COL}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_PW1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_PW2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_GAMMA,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SHORTCONV_CONV,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_SSM_CONV}
        },
        {
            tff::core::memory::LLM_TENSOR_SHORTCONV_INPROJ,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SHORTCONV_OUTPROJ,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        // NextN/MTP tensors are currently ignored (reserved for future MTP support)
        // These tensors only exist in the last layer(s) and are treated as output tensors
        {
            tff::core::memory::LLM_TENSOR_NEXTN_EH_PROJ,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_EMBED_TOKENS,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_ENORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_HNORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
    };
    static const std::unordered_map<tff::core::memory::ModelTensorType, std::pair<
        tff::core::model::ModelTensorLayerType, tff::core::graph::TffOpType> > LLM_LAYER_OP_INFOS_EXT = {
        {
            tff::core::memory::ModelTensorType::LLM_TENSOR_TOKEN_POS,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_MEM_REF}
        },
        {
            tff::core::memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_MEM_REF}
        },
        {
            tff::core::memory::LLM_TENSOR_TOKEN_EMBD,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_EMBD,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_TOKEN_EMBD_NORM,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_TOKEN_TYPES,
            {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {tff::core::memory::LLM_TENSOR_OUTPUT, {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}},
        {tff::core::memory::LLM_TENSOR_CLS, {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}},
        {tff::core::memory::LLM_TENSOR_CLS_OUT, {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}},
        {
            tff::core::memory::LLM_TENSOR_OUTPUT_NORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_OUTPUT_NORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_OUTPUT_NORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ROPE_FREQS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ROPE}
        },
        {
            tff::core::memory::LLM_TENSOR_ROPE_FACTORS_LONG,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ROPE}
        },
        {
            tff::core::memory::LLM_TENSOR_ROPE_FACTORS_SHORT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ROPE}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_QKV,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_DOWN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_UP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_DOWN_SHEXP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE_SHEXP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_UP_SHEXP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q_A,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_KV_A_MQA,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_KV_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_K_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_V_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_SINKS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_SCALE}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_FFN_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_FFN_DOWN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_FFN_UP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_FFN_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_FFN_DOWN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_FFN_UP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE_INP_SHEXP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE_INP,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_IN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_X,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_DT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_W1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_W2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_A1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_A2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_V1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_V2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_G1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_G2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_DECAY_W1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_DECAY_W2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_KEY,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_VALUE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_RECEPTANCE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_OUTPUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_KEY,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_RECEPTANCE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_VALUE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {tff::core::memory::LLM_TENSOR_FFN_ACT, {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_DIV}},
        {
            tff::core::memory::LLM_TENSOR_SSM_CONV1D,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_SSM_CONV}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_A,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_SSM_SCAN}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_DT_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_B_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_C_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_D,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SSM_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_X,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LN,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_LERP_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_CHANNEL_MIX_LERP_R,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_K_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_K_A,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_R_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_W,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_R,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_G,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_LERP_FUSED,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_DECAY,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_W0,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_A0,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_V0,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        {
            tff::core::memory::LLM_TENSOR_TIME_MIX_FIRST,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_RWKV_WKV6}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_NORM_2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_OUT_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_POST_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_POST_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_NORM_EXPS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_K_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_LAYER_OUT_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_Q_A_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_KV_A_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ATTN_SUB_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_SUB_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_FFN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_FFN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_DEC_ATTN_REL_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_ENC_ATTN_REL_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_DOWN_EXPS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT_ID}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_GATE_EXPS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT_ID}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_UP_EXPS,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT_ID}
        },
        {
            tff::core::memory::LLM_TENSOR_FFN_EXP_PROBS_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_ADD}
        },
        // altup / laurel (gemma 3n)
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_TOKEN_EMBD,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_MODEL_PROJ,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_PROJ_NORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_PROJ,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_UNEMBD_PROJ,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_INP_GATE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_PROJ,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_PER_LAYER_POST_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_CORRECT_COEF,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_CORRECT_SCALE,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_PREDICT_COEF,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_ROUTER,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_ALTUP_ROUTER_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_LAUREL_L,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_LAUREL_R,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_LAUREL_POST_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        // this tensor is loaded for T5, but never used
        {
            tff::core::memory::LLM_TENSOR_DEC_CROSS_ATTN_REL_B,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_NONE}
        },
        {tff::core::memory::LLM_TENSOR_CONV1D, {LLM_TENSOR_LAYER_INPUT, tff::core::graph::TffOpType::TFF_OP_IM2COL}},
        {
            tff::core::memory::LLM_TENSOR_POS_NET_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_NORM1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_NORM2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_CONV1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_IM2COL}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_CONV2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_IM2COL}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_Q,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_K,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_V,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_POS_NET_ATTN_OUT,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_DW,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_IM2COL}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_NORM,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_PW1,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_PW2,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_CONVNEXT_GAMMA,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_SHORTCONV_CONV,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_SSM_CONV}
        },
        {
            tff::core::memory::LLM_TENSOR_SHORTCONV_INPROJ,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_SHORTCONV_OUTPROJ,
            {LLM_TENSOR_LAYER_REPEATING, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        // NextN/MTP tensors are currently ignored (reserved for future MTP support)
        // These tensors only exist in the last layer(s) and are treated as output tensors
        {
            tff::core::memory::LLM_TENSOR_NEXTN_EH_PROJ,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_EMBED_TOKENS,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_ENORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_GET_ROWS}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_HNORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MUL_MAT}
        },
        {
            tff::core::memory::LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,
            {LLM_TENSOR_LAYER_OUTPUT, tff::core::graph::TffOpType::TFF_OP_MAP2CPU}
        },
    };
    static std::unordered_map<ModelMetaKV, uint32_t> LLM_SPECIAL_TOKENS = {
        {LLM_KV_TOKENIZER_BOS_ID, 1},
        {LLM_KV_TOKENIZER_EOS_ID, 2},
        {LLM_KV_TOKENIZER_EOT_ID, -1},
        {LLM_KV_TOKENIZER_EOM_ID, -1},
        {LLM_KV_TOKENIZER_UNK_ID, 0},
        {LLM_KV_TOKENIZER_SEP_ID, -1},
        {LLM_KV_TOKENIZER_PAD_ID, -1},
        {LLM_KV_TOKENIZER_MASK_ID, -1},
        {LLM_KV_TOKENIZER_FIM_PRE_ID, 13},
        {LLM_KV_TOKENIZER_FIM_SUF_ID, -1},
        {LLM_KV_TOKENIZER_FIM_MID_ID, -1},
        {LLM_KV_TOKENIZER_FIM_PAD_ID, -1},
        {LLM_KV_TOKENIZER_FIM_REP_ID, -1},
        {LLM_KV_TOKENIZER_FIM_SEP_ID, -1},
    };
    //
    inline constexpr std::array<std::string_view, static_cast<size_t>(VocabType::FF_VOCAB_TYPE_COUNT)>
    LLM_TOKENIZER_NAMES = {
        "NONE", "SPM", "BPE", "WPM", "UGM", "RWKV", "PLAMO2"
    };

    inline static std::string_view get_tokenizer_name(tff::core::model::VocabType type) {
        return LLM_TOKENIZER_NAMES.at(type);
    }

    static const std::unordered_map<std::string, VocabType> LLM_TOKENIZER_MODEL_VOCAB_TYPE = {
        {"gpt2", tff::core::model::VocabType::TFF_VOCAB_TYPE_BPE},
    };
    //
    static const std::unordered_map<ModelMetaKV, std::vector<std::string> > LLM_SPECIAL_TOKEN_STRING = {
        {
            LLM_KV_TOKENIZER_EOT_ID, std::vector<std::string>({
                "<|eot_id|>",
                "<|im_end|>",
                "<|end|>",
                "<end_of_turn>",
                "<|endoftext|>",
                "<EOT>",
                "_<EOT>",
                "<｜end▁of▁sentence｜>",
                "<end_of_utterance>",
            })
        },
        {
            LLM_KV_TOKENIZER_EOM_ID, std::vector<std::string>({
                "<|eom_id|>",
            })
        },
        {
            LLM_KV_TOKENIZER_FIM_PRE_ID, std::vector<std::string>({
                "<|fim_prefix|>",
                "<fim-prefix>",
                "<fim_prefix>",
                "<｜fim▁begin｜>",
                "<PRE>",
                "▁<PRE>",
                "<|code_prefix|>",
            })
        },
        {
            LLM_KV_TOKENIZER_FIM_SUF_ID, std::vector<std::string>({
                "<|fim_suffix|>",
                "<fim-suffix>",
                "<fim_suffix>",
                "<｜fim▁hole｜>",
                "<SUF>",
                "▁<SUF>",
                "<|code_suffix|>",
            })
        },
        {
            LLM_KV_TOKENIZER_FIM_MID_ID, std::vector<std::string>({
                "<|fim_middle|>",
                "<fim-middle>",
                "<fim_middle>",
                "<｜fim▁end｜>",
                "<MID>",
                "▁<MID>",
                "<|code_middle|>",
            })
        },
        {
            LLM_KV_TOKENIZER_FIM_PAD_ID, std::vector<std::string>({
                "<|fim_pad|>",
                "<fim-pad>",
                "<fim_pad>",
                "<PAD>",
            })
        },
        {
            LLM_KV_TOKENIZER_FIM_REP_ID, std::vector<std::string>({
                "<|fim_repo|>",
                "<|repo_name|>",
                "<fim-repo>",
                "<REPO>",
                "<reponame>",
            })
        },
        {
            LLM_KV_TOKENIZER_FIM_SEP_ID, std::vector<std::string>({
                "<|file_sep|>",
            })
        },
    };

    static const std::unordered_map<ModelArchitectureType, int32_t> TFF_MODEL_ARCHITECTURE_MAX_LAYER = {
        {tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA, LLAMA_MAX_LAYERS},
        {tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_UNKNOWN, -1},
    };
    static const std::unordered_map<tff::core::graph::TffOpType, const char *> TFF_OP_TYPE_MAP = {
        {tff::core::graph::TffOpType::TFF_OP_NONE, "none"},

        {tff::core::graph::TffOpType::TFF_OP_DUP, "dup"},
        {tff::core::graph::TffOpType::TFF_OP_ADD, "add"},
        {tff::core::graph::TffOpType::TFF_OP_ADD_ID, "add_id"},
        {tff::core::graph::TffOpType::TFF_OP_ADD1, "add1"},
        {tff::core::graph::TffOpType::TFF_OP_ACC, "acc"},
        {tff::core::graph::TffOpType::TFF_OP_SUB, "sub"},
        {tff::core::graph::TffOpType::TFF_OP_MUL, "mul"},
        {tff::core::graph::TffOpType::TFF_OP_DIV, "div"},
        {tff::core::graph::TffOpType::TFF_OP_SQR, "sqr"},
        {tff::core::graph::TffOpType::TFF_OP_SQRT, "sqrt"},
        {tff::core::graph::TffOpType::TFF_OP_LOG, "log"},
        {tff::core::graph::TffOpType::TFF_OP_SIN, "sin"},
        {tff::core::graph::TffOpType::TFF_OP_COS, "cos"},
        {tff::core::graph::TffOpType::TFF_OP_SUM, "sum"},
        {tff::core::graph::TffOpType::TFF_OP_SUM_ROWS, "sum_rows"},
        {tff::core::graph::TffOpType::TFF_OP_MEAN, "mean"},
        {tff::core::graph::TffOpType::TFF_OP_ARGMAX, "argmax"},
        {tff::core::graph::TffOpType::TFF_OP_COUNT_EQUAL, "count_equal"},
        {tff::core::graph::TffOpType::TFF_OP_REPEAT, "repeat"},
        {tff::core::graph::TffOpType::TFF_OP_REPEAT_BACK, "repeat_back"},
        {tff::core::graph::TffOpType::TFF_OP_CONCAT, "concat"},
        {tff::core::graph::TffOpType::TFF_OP_SILU_BACK, "silu_back"},
        {tff::core::graph::TffOpType::TFF_OP_NORM, "norm"}, // normalize
        {tff::core::graph::TffOpType::TFF_OP_RMS_NORM, "rms_norm"},
        {tff::core::graph::TffOpType::TFF_OP_RMS_NORM_BACK, "rms_norm_back"},
        {tff::core::graph::TffOpType::TFF_OP_GROUP_NORM, "group_norm"},
        {tff::core::graph::TffOpType::TFF_OP_L2_NORM, "l2_norm"},

        {tff::core::graph::TffOpType::TFF_OP_MUL_MAT, "mul_mat"},
        {tff::core::graph::TffOpType::TFF_OP_MUL_MAT_ID, "mul_mat_id"},
        {tff::core::graph::TffOpType::TFF_OP_OUT_PROD, "out_prod"},

        {tff::core::graph::TffOpType::TFF_OP_SCALE, "scale"},
        {tff::core::graph::TffOpType::TFF_OP_SET, "set"},
        {tff::core::graph::TffOpType::TFF_OP_CPY, "cpy"},
        {tff::core::graph::TffOpType::TFF_OP_CONT, "cont"},
        {tff::core::graph::TffOpType::TFF_OP_RESHAPE, "reshape"},
        {tff::core::graph::TffOpType::TFF_OP_VIEW, "view"},
        {tff::core::graph::TffOpType::TFF_OP_PERMUTE, "permute"},
        {tff::core::graph::TffOpType::TFF_OP_TRANSPOSE, "transpose"},
        {tff::core::graph::TffOpType::TFF_OP_GET_ROWS, "get_rows"},
        {tff::core::graph::TffOpType::TFF_OP_GET_ROWS_BACK, "get_rows_back"},
        {tff::core::graph::TffOpType::TFF_OP_SET_ROWS, "set_rows"},
        {tff::core::graph::TffOpType::TFF_OP_DIAG, "diag"},
        {tff::core::graph::TffOpType::TFF_OP_DIAG_MASK_INF, "diag_mask_inf"},
        {tff::core::graph::TffOpType::TFF_OP_DIAG_MASK_ZERO, "diag_mask_zero"},
        {tff::core::graph::TffOpType::TFF_OP_SOFT_MAX, "soft_max"},
        {tff::core::graph::TffOpType::TFF_OP_SOFT_MAX_BACK, "soft_max_back"},
        {tff::core::graph::TffOpType::TFF_OP_ROPE, "rope"},
        {tff::core::graph::TffOpType::TFF_OP_ROPE_BACK, "rope_back"},
        {tff::core::graph::TffOpType::TFF_OP_CLAMP, "clamp"},
        {tff::core::graph::TffOpType::TFF_OP_CONV_TRANSPOSE_1D, "conv_transpose_1d"},
        {tff::core::graph::TffOpType::TFF_OP_IM2COL, "im2col"},
        {tff::core::graph::TffOpType::TFF_OP_IM2COL_BACK, "im2col_back"},
        {tff::core::graph::TffOpType::TFF_OP_IM2COL_3D, "im2col_3d"},
        {tff::core::graph::TffOpType::TFF_OP_CONV_2D, "conv_2d"},
        {tff::core::graph::TffOpType::TFF_OP_CONV_3D, "conv_3d"},
        {tff::core::graph::TffOpType::TFF_OP_CONV_2D_DW, "conv_2d_dw"},
        {tff::core::graph::TffOpType::TFF_OP_CONV_TRANSPOSE_2D, "conv_transpose_2d"},
        {tff::core::graph::TffOpType::TFF_OP_POOL_1D, "pool_1d"},
        {tff::core::graph::TffOpType::TFF_OP_POOL_2D, "pool_2d"},
        {tff::core::graph::TffOpType::TFF_OP_POOL_2D_BACK, "pool_2d_back"},
        {tff::core::graph::TffOpType::TFF_OP_UPSCALE, "upscale"},
        {tff::core::graph::TffOpType::TFF_OP_PAD, "pad"},
        {tff::core::graph::TffOpType::TFF_OP_PAD_REFLECT_1D, "pad_reflect_1d"},
        {tff::core::graph::TffOpType::TFF_OP_ROLL, "roll"},
        {tff::core::graph::TffOpType::TFF_OP_ARANGE, "arange"},
        {tff::core::graph::TffOpType::TFF_OP_TIMESTEP_EMBEDDING, "timestep_embedding"},
        {tff::core::graph::TffOpType::TFF_OP_ARGSORT, "argsort"},
        {tff::core::graph::TffOpType::TFF_OP_LEAKY_RELU, "leaky_relu"},

        {tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT, "flash_attn_ext"},
        {tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_BACK, "flash_attn_back"},
        {tff::core::graph::TffOpType::TFF_OP_SSM_CONV, "ssm_conv"},
        {tff::core::graph::TffOpType::TFF_OP_SSM_SCAN, "ssm_scan"},
        {tff::core::graph::TffOpType::TFF_OP_WIN_PART, "win_part"},
        {tff::core::graph::TffOpType::TFF_OP_WIN_UNPART, "win_unpart"},
        {tff::core::graph::TffOpType::TFF_OP_GET_REL_POS, "get_rel_pos"},
        {tff::core::graph::TffOpType::TFF_OP_ADD_REL_POS, "add_rel_pos"},
        {tff::core::graph::TffOpType::TFF_OP_RWKV_WKV6, "rwkv_wkv6"},
        {tff::core::graph::TffOpType::TFF_OP_GATED_LINEAR_ATTN, "gated_linear_attn"},
        {tff::core::graph::TffOpType::TFF_OP_RWKV_WKV7, "rwkv_wkv7"},

        {tff::core::graph::TffOpType::TFF_OP_UNARY, "unary"},

        {tff::core::graph::TffOpType::TFF_OP_MAP_CUSTOM1, "map_custom1"},
        {tff::core::graph::TffOpType::TFF_OP_MAP_CUSTOM2, "map_custom2"},
        {tff::core::graph::TffOpType::TFF_OP_MAP_CUSTOM3, "map_custom3"},

        {tff::core::graph::TffOpType::TFF_OP_CUSTOM, "custom"},

        {tff::core::graph::TffOpType::TFF_OP_CROSS_ENTROPY_LOSS, "cross_entropy_loss"},
        {tff::core::graph::TffOpType::TFF_OP_CROSS_ENTROPY_LOSS_BACK, "cross_entropy_loss_back"},
        {tff::core::graph::TffOpType::TFF_OP_OPT_STEP_ADAMW, "opt_step_adamw"},
        {tff::core::graph::TffOpType::TFF_OP_OPT_STEP_SGD, "opt_step_sgd"},

        {tff::core::graph::TffOpType::TFF_OP_GLU, "glu"},
        {tff::core::graph::TffOpType::TFF_OP_MAP2CPU, "map2cpu"},
        {tff::core::graph::TffOpType::TFF_OP_MEM_CPY, "memcpy"},
        {tff::core::graph::TffOpType::TFF_OP_MEM_REF, "mem_ref"},
        {tff::core::graph::TffOpType::TFF_OP_EMBEDDING, "embedding"},
        //quant
        {tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8, "quantize_q8"},
        {tff::core::graph::TffOpType::TFF_OP_DEQUANTIZE_Q8, "dequantize_q8"},
        {tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_MATMUL, "quant_q8_matmul"},
        {tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_RESHAPE, "quantize_q8_reshape"},
        {tff::core::graph::TffOpType::TFF_OP_DEQUANTIZE_Q8_RESHAPE, "dequantize_q8_reshape"},
        //
        {tff::core::graph::TffOpType::TFF_OP_PRE_ROPE_TABLE, "pre_rope_table"},
    };

    //op fuse model
    static const std::unordered_map<tff::core::graph::TffOpType, std::vector<tff::core::graph::TffOpType> >
    TFF_OP_FUSE_MODEL = {
        {
            tff::core::graph::TffOpType::TFF_OP_ROPE,
            {tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT}
        }, //flash attention与rope算子融合
        {tff::core::graph::TffOpType::TFF_OP_RMS_NORM, {graph::TffOpType::TFF_OP_MUL, graph::TffOpType::TFF_OP_ADD}},
        {tff::core::graph::TffOpType::TFF_OP_RMS_NORM, {graph::TffOpType::TFF_OP_MUL}},
    };

    template<typename T>
    struct is_vector : std::false_type {
    };

    template<typename T>
    struct is_vector<std::vector<T> > : std::true_type {
    };

    template<typename T>
    constexpr bool is_vector_v = is_vector<T>::value;

    template<typename To, typename From>
    std::optional<To> try_convert(const From &value) {
        if constexpr (std::is_same_v<To, From>) {
            return value;
        } else if constexpr (std::is_arithmetic_v<To> && std::is_arithmetic_v<From>) {
            return static_cast<To>(value);
        } else if constexpr (std::is_same_v<To, std::string> && std::is_constructible_v<std::string, From>) {
            return std::string(value);
        } else {
            return std::nullopt;
        }
    }

    //
    //
    template<class Key, class ValueType, class DataType>
    std::vector<DataType> get_general_value(const std::string key_value,
                                            const std::shared_ptr<tff::core::model::ModelContext> &ctx) {
        std::vector<DataType> result;


        auto kv_it = ctx->_kv.find(key_value);
        if (kv_it == ctx->_kv.end()) {
            return result;
        }

        const auto &kvval = kv_it->second;

        std::visit([&result](const auto &val) {
            using T = std::decay_t<decltype(val)>;

            if constexpr (std::is_same_v<T, std::vector<DataType> >) {
                result = val;
            }
            else if constexpr (is_vector_v<T>) {
                using U = typename T::value_type;
                for (const auto &item: val) {
                    if constexpr (std::is_convertible_v<U, DataType>) {
                        result.push_back(static_cast<DataType>(item));
                    } else if constexpr (requires { try_convert<DataType>(item); }) {
                        if (auto cvt = try_convert<DataType>(item)) {
                            result.push_back(*cvt);
                        }
                    }
                }
            }
            else if constexpr (std::is_same_v<T, ModelContext::BasicType>) {
                std::visit([&result](const auto &inner) {
                    using U = std::decay_t<decltype(inner)>;
                    if constexpr (std::is_convertible_v<U, DataType>) {
                        result.push_back(static_cast<DataType>(inner));
                    } else if constexpr (requires { try_convert<DataType>(inner); }) {
                        if (auto cvt = try_convert<DataType>(inner)) {
                            result.push_back(*cvt);
                        }
                    }
                }, val);
            }
            else if constexpr (!is_vector_v<T> && !std::is_same_v<T, ModelContext::BasicType>) {
                if constexpr (std::is_convertible_v<T, DataType>) {
                    result.push_back(static_cast<DataType>(val));
                } else if constexpr (requires { try_convert<DataType>(val); }) {
                    if (auto cvt = try_convert<DataType>(val)) {
                        result.push_back(*cvt);
                    }
                }
            }
        }, kvval);
        return result;
    }

    static inline bool is_general_kv(const std::string &kv_value) {
        if (kv_value.find("%") != kv_value.npos) {
            return false;
        }
        return true;
    }

    //
    template<class Key, class ValueType, class DataType>
    std::vector<DataType> get_value(Key key_value,
                                    const std::shared_ptr<tff::core::model::ModelContext> &ctx) {
        std::vector<DataType> result;
        std::string key_name(
            tff::core::global::LLM_KV_NAMES.find(key_value)->second);
        if (!is_general_kv(key_name)) {
            std::string arch_name(
                tff::core::global::LLM_KV_NAMES.find(
                    tff::core::model::ModelMetaKV::LLM_KV_GENERAL_ARCHITECTURE)->second);
            arch_name = get_general_value<Key, std::string, std::string>(arch_name, ctx)[0];
            char buffer[256];
            std::sprintf(buffer, key_name.c_str(), arch_name.c_str());
            key_name = buffer;
        }
        result = get_general_value<Key, ValueType, DataType>(key_name, ctx);
        if (result.empty()) {
            result.push_back(0);
        }
        return result;
    }

    //
    template<typename T>
    constexpr const char *get_type_suffix() {
        if constexpr (std::is_same_v<T, float>) return "f32";
        else if constexpr (std::is_same_v<T, double>) return "f64";
        else if constexpr (std::is_same_v<T, int8_t>) return "i8";
        else if constexpr (std::is_same_v<T, uint8_t>) return "u8";
        else if constexpr (std::is_same_v<T, int16_t>) return "i16";
        else if constexpr (std::is_same_v<T, int32_t>) return "i32";
        else if constexpr (std::is_same_v<T, int64_t>) return "i64";
        else if constexpr (std::is_same_v<T, half>) return "fp16";
        else if constexpr (std::is_same_v<T, Q8_0>) return "q8_0";
        else return "unknown";
    }
}
#endif //TFFINFER_MODELGLOBALVAR_H
