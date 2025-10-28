//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_TENSOR_H
#define TFFINFER_TENSOR_H

#include <numeric>
#include <utility>

#include "Memory.h"
#include "BaseDefine.h"

namespace tff::core::memory {

    class Tensor {
    public:
        Tensor(const tff::core::memory::DataType data_type, std::vector<int64_t> shapes, bool use_external = false,
               std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> alloc =
                       nullptr) : _use_external(false), _data_type(data_type), _shape(std::move(shapes)),
                                  _allocator(std::move(alloc)) {
            this->set_dims(_shape.size());
            for (size_t i = 0; i < this->_shape.size(); ++i) {
                this->set_shape(this->_shape[i], i);
            }
            this->set_data_type(data_type);
            if (!use_external) {
                _buffer = std::make_shared<tff::core::memory::Memory>(
                    type_traits_auto[data_type]._type_size * std::accumulate(
                        _shape.begin(), _shape.end(), 1, std::multiplies<int64_t>()), nullptr, use_external,
                    _allocator);
                _buffer->allocate();
            }
        }

        ~Tensor() = default;

    public:
        //
        [[nodiscard]] inline size_t get_row_size() const {
            return _type_size * this->_shape[0] / type_traits_auto[_data_type]._blck_size;
        }

        [[nodiscard]] inline size_t get_bytes() const {
            size_t nbytes = 0;
            const size_t blck_size = type_traits_auto[_data_type]._blck_size;
            if (blck_size == 1) {
                for (int i = 0; i < _n_dims; ++i) {
                    nbytes += (this->_shape[i]) * this->_shape_bytes[i];
                }
            } else {
                nbytes = this->_shape[0] * this->_shape_bytes[0] / blck_size;
                for (int i = 1; i < _n_dims; ++i) {
                    nbytes += (this->_shape[i] - 1) * this->_shape_bytes[i];
                }
            }

            return nbytes;
        }

        inline void set_dims(const size_t &n_dims) {
            this->_n_dims = n_dims;
        }

        //
        inline void set_shape(const size_t &n_dims, const size_t &index) {
            this->_shape[index] = n_dims;
        }

        //
        inline void set_data_type(const DataType &data_type) {
            this->_data_type = data_type;
            _type_size = memory::type_traits_auto[tff::core::memory::DataType(data_type)]._type_size;
            _blk_size = memory::type_traits_auto[tff::core::memory::DataType(data_type)]._blck_size;

            _shape_bytes[0] = _type_size;
            _shape_bytes[1] = _shape_bytes[0] * (_shape[0] / _blk_size);
            for (int j = 2; j < _n_dims; ++j) {
                _shape_bytes[j] = _shape_bytes[j - 1] * _shape[j - 1];
            }
        }

        //
        void set_buffer_data(void *data, const size_t &buffer_size);
        //
        [[nodiscard]] inline tff::core::memory::ModelTensorType get_tensor_type() const {
            return this->_tensor_type;
        }

    private:
        bool _use_external = false;
        size_t _n_dims{};
        tff::core::memory::DataType _data_type;
        tff::core::memory::ModelTensorType _tensor_type;
        size_t _type_size{};
        int64_t _blk_size{};
        std::vector<int64_t> _shape;
        std::array<int64_t, 4> _shape_bytes;
        std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> _allocator;
        std::shared_ptr<tff::core::memory::Memory> _buffer;
    };
}

#endif //TFFINFER_TENSOR_H
