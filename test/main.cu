//
// Created by nkk on 2025/9/28.
//
#include <stdio.h>
#include <iostream>
#include "device/cuda/cudaInc.h"
#include "runtime/LLMInferRuntime.h"

static void rtrim(std::string &s) {
    s.erase(
        std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(),
        s.end()
    );
}

int main(int argc, char *argv[]) {
    std::string model_config_file_path(argv[1]);
    std::string model_file(argv[2]);
    tff::core::runtime::LLMInferRuntime llm_runtime;
    llm_runtime.init_device();
    tff::core::model::ModelConfig cfg;
    cfg._use_mmap = true;
    cfg._use_f16 = true;
    cfg._is_fuse_op = true;
    cfg._kv_data_type = tff::core::memory::DataType::TFF_DATA_TYPE_F16;
    cfg._rope_freq_base = 1000000;
    cfg._rope_freq_scale = 1;
    const int max_batches = 1;
    llm_runtime.load_model_config(model_config_file_path, cfg);
    std::vector<std::string> model_files;
    model_files.push_back(model_file);
    llm_runtime.load_model(model_files, cfg);
    bool bRet = llm_runtime.init_runtime_context();
    if (!bRet) {
        printf("llm_runtime.init_runtime_context() failed\n");
        return -1;
    }

    std::string prompt0 = "你好，我是一个AI芯片公司的高性能计算程序员，主要负责大模型推理框架的迭代和优化，年底了，请帮我写一份年终总结大纲。要求尽可能全面。";
    std::string prompt1 = "What is your name";
    std::string prompt2 = "Hello my name is";
    std::vector<std::string> prompt_pre{prompt0, prompt1, prompt2};
    std::vector<std::string> names{"", "", "Wang Peng"};
    std::vector<std::string> prompt_batches;
    prompt_batches.resize(max_batches);
    for (int i = 0; i < max_batches; i++) {
        prompt_batches[i] = prompt_pre[i] + " " + names[i];
        rtrim(prompt_batches[i]);
    }
    std::string respone_str = "";
    const int n_predict = 256;
    llm_runtime.encode(prompt_batches);
    llm_runtime.prefill();
    llm_runtime.decode(n_predict, respone_str);
    return 0;
}
