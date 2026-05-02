//
// Created by nkk on 2025/9/28.
//
#include <stdio.h>
#include <iostream>
#include "device/cuda/cudaInc.h"
#include "runtime/InferRuntime.h"

static void rtrim(std::string &s) {
    s.erase(
        std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(),
        s.end()
    );
}

int main(int argc, char *argv[]) {
    std::string model_file(argv[1]);
    //std::string model_config_file_path(argv[2]);
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
    bool bRet = llm_runtime.init_runtime_context();
    if (!bRet) {
        printf("llm_runtime.init_runtime_context() failed\n");
        return -1;
    }

    std::string prompt0 = "你好，我是一个AI芯片公司的高性能计算程序员，主要负责大模型推理框架的迭代和优化，年底了，请帮我写一份年终总结大纲。要求尽可能全面。";
    std::string prompt2 = "Hello my name is NKK";
    std::string prompt4 = "你好，我是一个AI芯片公司的高性能计算程序员，主要负责大模型推理框架的迭代和优化";
    std::string prompt3 =R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User:
Recommend a nice restaurant in the area.
Assistant:
I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User:
Who is Richard Feynman?
Assistant:
Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
)";
    std::vector<std::string> prompt_pre{prompt0, prompt2, prompt4, prompt0};
    std::vector<std::string> names{"", "", "Wang Peng"};
    std::vector<std::string> prompt_batches;
    prompt_batches.resize(max_batches);
    for (int i = 0; i < max_batches; i++) {
        prompt_batches[i] = prompt_pre[i];
        rtrim(prompt_batches[i]);
    }
    std::vector<std::string> respone_str;
    constexpr int n_predict = 64;
    llm_runtime.encode(prompt_batches);
    llm_runtime.infer(n_predict, respone_str);
    return 0;
}
