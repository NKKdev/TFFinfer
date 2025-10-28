//
// Created by nkk on 2025/9/28.
//
#include <stdio.h>
#include <iostream>
#include "device/cuda/cudaInc.h"
#include "model/LLMModel.h"
int main(int argc,char *argv[]) {
    std::string model_config_file_path(argv[1]);
    std::string model_file(argv[2]);
    tff::core::model::LLMModel llm_model;
    tff::core::model::ModelConfig cfg;
    llm_model.load_model_config(model_config_file_path, cfg);
    std::vector<std::string> model_files;
    model_files.push_back(model_file);
    llm_model.load_model(model_files, cfg);
    return 0;
}