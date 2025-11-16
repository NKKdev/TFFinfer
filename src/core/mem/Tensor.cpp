//
// Created by nkk on 2025/9/28.
//

#include "Tensor.h"

void tff::core::memory::Tensor::set_buffer_data(void *data, const size_t &buffer_size) {
    _use_external = true;
    this->_buffer = std::make_shared<tff::core::memory::Memory>(buffer_size, data, _use_external);
}
