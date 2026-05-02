//
// Created by nkk on 2026/5/2.
//
#include <stdio.h>
#include <iostream>
#include "runtime/InferRuntime.h"

int main(int argc, char *argv[]) {
    std::string model_file(argv[1]);
    tff::core::runtime::LLMInferRuntime llm_runtime;
    tff::core::model::ModelConfig cfg;
    cfg._use_mmap = true;
    cfg._use_f16 = true;
    cfg._is_fuse_op = true;
    cfg._kv_data_type = tff::core::memory::DataType::TFF_DATA_TYPE_F16;
    cfg._rope_freq_base = 1000000;
    cfg._rope_freq_scale = 1;
    cfg._n_output = 1;
    const int max_batches = 1;
    std::vector<std::string> model_files;
    model_files.push_back(model_file);
    llm_runtime.load_model(model_files, cfg);

    return 0;
}
