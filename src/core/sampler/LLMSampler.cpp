//
// Created by nkk on 2026/2/26.
//

#include "LLMSampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

#include "mem/Tensor.h"
namespace tff::core::sampling {
    int32_t LLMSampler::sample(
        int64_t vocab_size,
        float *logits,
        const SamplingConfig &config,
        std::mt19937 &rng) {
        return sample_greedy(logits, vocab_size);
    }

    void LLMSampler::apply_repetition_penalty(
        float *logits, size_t vocab_size,
        const std::vector<int32_t> &prompt,
        const std::vector<int32_t> &output,
        float penalty) {
        std::unordered_set<int32_t> seen_tokens(prompt.begin(), prompt.end());
        seen_tokens.insert(output.begin(), output.end());
        for (int32_t token: seen_tokens) {
            if (token >= 0 && token < static_cast<int32_t>(vocab_size)) {
                if (logits[token] > 0) logits[token] /= penalty;
                else logits[token] *= penalty;
            }
        }
    }

    void LLMSampler::apply_temperature(float *logits, size_t vocab_size, float temp) {
        float inv_temp = 1.0f / temp;
        for (size_t i = 0; i < vocab_size; ++i) {
            logits[i] *= inv_temp;
        }
    }

    int32_t LLMSampler::sample_greedy(const float *logits, size_t vocab_size) {
        return static_cast<int32_t>(std::max_element(logits, logits + vocab_size) - logits);
    }

    int32_t LLMSampler::sample_multinomial(const float *logits, size_t vocab_size, std::mt19937 &rng) {
        std::vector<double> probs(logits, logits + vocab_size);
        double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
        if (sum == 0.0) return 0; // fallback
        for (auto &p: probs) p /= sum;
        std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
        return dist(rng);
    }
}
