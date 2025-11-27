//
// Created by nkk on 2025/9/28.
//
#include <stdio.h>
#include <iostream>
#include "device/cuda/cudaInc.h"
#include "runtime/LLMInferRuntime.h"
int main(int argc,char *argv[]) {
    std::string model_config_file_path(argv[1]);
    std::string model_file(argv[2]);
    tff::core::runtime::LLMInferRuntime llm_runtime;
    llm_runtime.init_device();
    tff::core::model::ModelConfig cfg;
    cfg._use_mmap = true;
    llm_runtime.load_model_config(model_config_file_path, cfg);
    std::vector<std::string> model_files;
    model_files.push_back(model_file);
    llm_runtime.load_model(model_files, cfg);
    llm_runtime.init_runtime_context();

    std::string prompt = "Hello my name is";
    std::string respone_str = "";
    const int n_predict = 256;
    llm_runtime.prefill(std::vector<std::string>{prompt});
    //llm_runtime.decode(n_predict, respone_str);
    return 0;
}