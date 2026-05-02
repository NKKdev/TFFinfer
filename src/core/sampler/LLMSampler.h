//
// Created by nkk on 2026/2/26.
//

#ifndef TFFINFER_LLMSAMPLER_H
#define TFFINFER_LLMSAMPLER_H
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

namespace tff::core::memory {
    class Tensor;
}

namespace tff::core::sampling {
#define FOR_EACH_STRATEGY(X) \
    X(TFF_SAMPLE_STRATEGY_DEFAULT, "default")\
    X(TFF_SAMPLE_STRATEGY_UNKNOWN, "unknown")\
    X(TFF_SAMPLE_STRATEGY_GREEDY, "greedy")\
    X(TFF_SAMPLE_STRATEGY_RANDOM, "random")\
    X(TFF_SAMPLE_STRATEGY_TOP_K, "top_k")\
    X(TFF_SAMPLE_STRATEGY_TOP_P, "top_p")

    /**
     * @brief 采样策略类型
     */
    enum SampleStrategyType {
#define DEFINE_ENUM(name, type_str) name,
        FOR_EACH_STRATEGY(DEFINE_ENUM)
#undef DEFINE_ENUM
        TFF_SAMPLE_STRATEGY_COUNT
    };

    constexpr const char *to_string(const SampleStrategyType arch_type) {
#define CASE_STR(name, str) case SampleStrategyType::name: return str;
        switch (arch_type) {
            FOR_EACH_STRATEGY(CASE_STR)
            default: return "invalid";
        }
#undef CASE_STR
    }

    constexpr SampleStrategyType from_string(std::string_view enum_str) {
#define CASE_STR(name, str) if (enum_str == str) return SampleStrategyType::name;
        FOR_EACH_STRATEGY(CASE_STR)
#undef CASE_STR
        return SampleStrategyType::TFF_SAMPLE_STRATEGY_UNKNOWN;
    }
#undef FOR_EACH_STRATEGY
    /**
     * @brief 采样配置
     */
    struct SamplingConfig {
        SampleStrategyType _type = SampleStrategyType::TFF_SAMPLE_STRATEGY_GREEDY;
        float _temperature = 1.0f;
        int _top_k = 0;
        float _top_p = 1.0f;
        float _min_p = 0.0f;
        float _repetition_penalty = 1.0f;
        size_t _seed = 0;
    };
    /**
     * @brief LLM采样器
     */
    class LLMSampler {
    public:
        /**
         * @brief 采样
         * @param logits 采样输入
         * @param config 采样配置
         * @param rng 随机数生成器
         * @return 采样结果
         */
        static std::int32_t sample(
            int64_t vocab_size,
            float *logits, const SamplingConfig &config, std::mt19937 &rng);
        /**todo
         * @brief 批量采样
         * @param logits_tensors 采样输入
         * @param all_prompt_tokens 提示词
         * @param all_output_tokens 输出词
         * @param config 采样配置
         * @return 采样结果
         */
        static std::vector<int32_t> sample_batch(
            const std::vector<std::shared_ptr<tff::core::memory::Tensor>>& logits_tensors,
            const std::vector<std::vector<int32_t>>& all_prompt_tokens,
            const std::vector<std::vector<int32_t>>& all_output_tokens,
            const SamplingConfig& config);

    private:
        /**todo
         * @brief 应用重复惩罚
         * @param logits 采样输入
         * @param vocab_size 词表大小
         * @param prompt_tokens 提示词
         * @param output_tokens 输出词
         * @param penalty 重复惩罚
         */
        static void apply_repetition_penalty(
            float* logits, size_t vocab_size,
            const std::vector<int32_t>& prompt_tokens,
            const std::vector<int32_t>& output_tokens,
            float penalty);
        /**todo
         * @brief 应用温度
         * @param logits 采样输入
         * @param vocab_size 词表大小
         * @param temp 温度
         */
        static void apply_temperature(float* logits, size_t vocab_size, float temp);
        /**
         * @brief 贪心采样
         * @param logits 采样输入
         * @param vocab_size 词表大小
         * @return 采样结果
         */
        static int32_t sample_greedy(const float* logits, size_t vocab_size);
        /**todo
         * @brief 多项式采样
         * @param logits 采样输入
         * @param vocab_size 词表大小
         * @param rng 随机数生成器
         * @return 采样结果
         */
        static int32_t sample_multinomial(const float* logits, size_t vocab_size, std::mt19937& rng);

    };
}

#endif //TFFINFER_LLMSAMPLER_H
