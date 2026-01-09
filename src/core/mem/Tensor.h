//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_TENSOR_H
#define TFFINFER_TENSOR_H

#include <numeric>
#include <utility>
#include <mutex>
#include "Memory.h"
#include "BaseDefine.h"
#include "global/GlobalDefine.h"

namespace tff::core::memory {
    class Tensor :public std::enable_shared_from_this<Tensor>{
    public:
        struct TensorCompare {
            bool operator()(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) const {
                return a->get_priority() < b->get_priority();
            }
        };

    public:
        Tensor(const std::shared_ptr<Tensor> &other, bool _use_external = false) noexcept {
            release();
            _use_external = other->_use_external;
            _external_memory_index = other->_external_memory_index;
            _data_type = other->_data_type;
            _tensor_type = other->_tensor_type;
            _type_size = other->_type_size;
            _blk_size = other->_blk_size;
            _n_dims = other->_n_dims;
            _shape = other->_shape;
            _strides = other->_strides;
            // _allocator = other->_allocator;
            // if (!_use_external) {
            //     auto total_bytes = std::accumulate(
            //                            _shape.begin(), _shape.end(), 1ULL, std::multiplies<uint64_t>()
            //                        ) * _type_size;
            //     _buffer = std::make_shared<tff::core::memory::Memory>(
            //         total_bytes, nullptr, _use_external, _allocator
            //     );
            //     _buffer->allocate();
            //     _is_allocated = true;
            // } else {
            //     if (other->_buffer) {
            //         _buffer = std::make_shared<tff::core::memory::Memory>(
            //             other->_buffer->byte_size(), other->_buffer->ptr(), true, _allocator
            //         );
            //     }
            //     _is_allocated = other->_is_allocated;
            // }
        }
        Tensor(const int n_dim = 4, const tff::core::memory::DataType data_type = tff::core::memory::DataType::TFF_DATA_TYPE_UNKNOWN,
               std::array<int64_t, MAX_TENSOR_DIM> shapes = std::array<int64_t, MAX_TENSOR_DIM>(),
               bool use_external = true,
               std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> alloc =
                       nullptr) : _is_allocated(false), _use_external(use_external), _data_type(data_type),
                                  _shape(std::move(shapes)),
                                  _allocator(std::move(alloc)) {
            if (_shape.empty()) {
                tff::log::Logger::error("tensor shape is invalid!!");
                return;
            }
            this->set_dims(n_dim);
            for (size_t i = 0; i < this->_shape.size(); ++i) {
                this->set_shape(this->_shape[i], i);
            }
            this->set_data_type(data_type);
            if (!use_external) {
                this->allocate();
            }
        }
        Tensor(const tff::core::memory::DataType data_type = tff::core::memory::DataType::TFF_DATA_TYPE_UNKNOWN,
               std::array<int64_t, MAX_TENSOR_DIM> shapes = std::array<int64_t, MAX_TENSOR_DIM>(),
               bool use_external = true,
               std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> alloc =
                       nullptr) : _is_allocated(false), _use_external(use_external), _data_type(data_type),
                                  _shape(std::move(shapes)),
                                  _allocator(std::move(alloc)) {
            if (_shape.empty()) {
                tff::log::Logger::error("tensor shape is invalid!!");
                return;
            }
            this->set_dims(shapes.size());
            for (size_t i = 0; i < this->_shape.size(); ++i) {
                this->set_shape(this->_shape[i], i);
            }
            this->set_data_type(data_type);
            if (!use_external) {
                this->allocate();
            }
        }

        Tensor &operator=(const Tensor &other) {
            if (this == &other) {
                return *this;
            }
            release();

            _use_external = other._use_external;
            _external_memory_index = other._external_memory_index;
            _data_type = other._data_type;
            _tensor_type = other._tensor_type;
            _type_size = other._type_size;
            _blk_size = other._blk_size;
            _n_dims = other._n_dims;
            _shape = other._shape;
            _strides = other._strides;
            _allocator = other._allocator;
            if (!_use_external) {
                auto total_bytes = std::accumulate(
                                       _shape.begin(), _shape.end(), 1ULL, std::multiplies<uint64_t>()
                                   ) * _type_size;
                _buffer = std::make_shared<tff::core::memory::Memory>(
                    total_bytes, nullptr, _use_external, _allocator
                );
                _buffer->allocate();
                _is_allocated = true;
                if (other._buffer && other._buffer->ptr() && _buffer->ptr()) {
                    _allocator->memcopy(_buffer->ptr(), other._buffer->ptr(), total_bytes);
                }
            } else {
                if (other._buffer) {
                    _buffer = std::make_shared<tff::core::memory::Memory>(
                        other._buffer->byte_size(), other._buffer->ptr(), true, _allocator
                    );
                }
                _is_allocated = other._is_allocated;
            }
            return *this;
        }

        Tensor &operator=(Tensor &&other) noexcept {
            if (this == &other) {
                return *this;
            }
            release();

            _use_external = other._use_external;
            _is_allocated = other._is_allocated;
            _data_type = other._data_type;
            _tensor_type = other._tensor_type;
            _type_size = other._type_size;
            _blk_size = other._blk_size;
            _n_dims = other._n_dims;
            _shape = std::move(other._shape);
            _strides = other._strides;
            _allocator = std::move(other._allocator);
            _buffer = std::move(other._buffer);
            other._is_allocated = false;
            other._use_external = false;
            other._n_dims = 0;
            return *this;
        }

        // bool operator==(const Tensor& other) const {
        //     if (_data_type != other._data_type ||
        //         _shape != other._shape ||
        //         _use_external != other._use_external) {
        //         return false;
        //         }
        //     if (!_is_allocated && !other._is_allocated) {
        //         return true;
        //     }
        //
        //     if (!_is_allocated || !other._is_allocated) {
        //         return false;
        //     }
        //     size_t this_bytes = get_bytes();
        //     size_t other_bytes = other.get_bytes();
        //     if (this_bytes != other_bytes) {
        //         return false;
        //     }
        //     if (!_buffer || !_buffer->ptr() || !other._buffer || !other._buffer->ptr()) {
        //         return false;
        //     }
        //     return std::memcmp(_buffer->ptr(), other._buffer->ptr(), this_bytes) == 0;
        // }


        ~Tensor() {
        }

    public:
        //
        inline void allocate() {
            std::lock_guard<std::mutex> lock(this->_mutex);
            if (!_use_external) {
                _buffer = std::make_shared<tff::core::memory::Memory>(
                    type_traits_auto[this->_data_type]._type_size * std::accumulate(
                        _shape.begin(), _shape.end(), 1, std::multiplies<int64_t>()), nullptr, _use_external,
                    _allocator);
                _buffer->allocate();
                _is_allocated = true;
            }
        }

        //
        [[nodiscard]] inline size_t get_row_size() const {
            return _type_size * this->_shape[0] / type_traits_auto[_data_type]._blck_size;
        }

        [[nodiscard]] inline size_t get_bytes() const {
            size_t nbytes = 0;
            const size_t blck_size = type_traits_auto[_data_type]._blck_size;
            if (blck_size == 1) {
                for (int i = _n_dims - 1; i < _n_dims; ++i) {
                    nbytes += (this->_shape[i]) * this->_strides[i];
                }
            } else {
                nbytes = this->_shape[0] * this->_strides[0] / blck_size;
                for (int i = _n_dims - 1; i < _n_dims; ++i) {
                    nbytes += (this->_shape[i] - 1) * this->_strides[i];
                }
            }

            return nbytes;
        }

        inline void set_dims(const size_t &n_dims) {
            this->_n_dims = n_dims;
        }

        //
        inline std::array<int64_t, MAX_TENSOR_DIM> &get_shape() {
            return _shape;
        }

        //
        inline std::array<int64_t, MAX_TENSOR_DIM> &get_strides() {
            return this->_strides;
        }

        //
        inline void set_shape(const size_t &n_dims, const size_t &index) {
            this->_shape[index] = n_dims;
        }

        //
        inline void set_shape(const std::array<int64_t, MAX_TENSOR_DIM> &shape) {
            this->_shape = shape;
            this->set_dims(this->_shape.size());
        }

        //
        inline DataType get_data_type() const {
            return this->_data_type;
        }

        //
        inline void set_data_type(const DataType &data_type) {
            this->_data_type = data_type;
            _type_size = memory::type_traits_auto[tff::core::memory::DataType(data_type)]._type_size;
            _blk_size = memory::type_traits_auto[tff::core::memory::DataType(data_type)]._blck_size;
            if (this->_strides.empty()) {
                tff::log::Logger::error("_strides is empty!!");
                return;
            }
            _strides[0] = _type_size;
            if (this->_strides.size() > 1) {
                _strides[1] = _strides[0] * (_shape[0] / _blk_size);
            }
            for (int j = 2; j < _shape.size(); ++j) {
                _strides[j] = _strides[j - 1] * _shape[j - 1];
            }
        }

        //
        inline void set_buffer_data(void *data, const size_t &buffer_size, size_t mem_buffer_index = -1) {
            std::lock_guard<std::mutex> lock(this->_mutex);
            if (this->_buffer != nullptr) {
                tff::log::Logger::error("current tensor already has buffer!!");
                return;
            }
            _use_external = true;
            this->_external_memory_index = mem_buffer_index;
            this->_buffer = std::make_shared<tff::core::memory::Memory>(buffer_size, data, _use_external);
        }

        //
        inline size_t get_external_memory_index() const {
            std::lock_guard<std::mutex> lock(this->_mutex);
            return _external_memory_index;
        }
        inline void set_external_memory_index(const size_t &mem_offset) {
            std::lock_guard<std::mutex> lock(this->_mutex);
            this->_external_memory_index = mem_offset;
        }

        //
        [[nodiscard]] inline tff::core::memory::ModelTensorType get_tensor_type() const {
            return this->_tensor_type;
        }

        inline void set_tensor_type(const tff::core::memory::ModelTensorType &tensor_type) {
            this->_tensor_type = tensor_type;
        }

        //
        inline void release() {
            if (_buffer) {
                _allocator->release(_buffer->ptr());
            }
            _type_size = 0;
            _blk_size = 0;
        }

        //
        inline std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> &get_allocator() {
            return _allocator;
        }

        //
        inline void set_allocator(const std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> &allocator) {
            _allocator = allocator;
        }

        //
        [[nodiscard]] inline bool is_allocated() const {
            return _is_allocated;
        }

        //
        inline std::shared_ptr<tff::core::memory::Memory> &get_buffer() {
            std::lock_guard<std::mutex> lock(this->_mutex);
            return _buffer;
        }

        //
        [[nodiscard]] inline int get_priority() const {
            return _priority;
        }

        //
        inline void set_priority(const int &priority) {
            _priority = priority;
        }
        //
        inline int get_ref_count() const {
            try {
                tff::log::Logger::info("current ref count: %d ", shared_from_this().use_count());
                return shared_from_this().use_count();
            } catch (const std::bad_weak_ptr& e) {
                return -1;
            }
        }

        //
        template<typename T, typename... Args>
        inline T &at(Args... indices) {
            if (sizeof(T) != _type_size) {
                throw std::runtime_error("Tensor::get<T>(): sizeof(T) != element size");
            }
            if (!_buffer || !_buffer->ptr()) {
                throw std::runtime_error("Tensor buffer is null");
            }
            return *reinterpret_cast<T *>(_buffer->ptr() + compute_offset(indices...));
        }

        template<typename T, typename... Args>
        inline const T &at(Args... indices) const {
            if (sizeof(T) != _type_size) {
                throw std::runtime_error("Tensor::get<T>(): sizeof(T) != element size");
            }
            if (!_buffer || !_buffer->ptr()) {
                throw std::runtime_error("Tensor buffer is null");
            }
            return *reinterpret_cast<const T *>(_buffer->ptr() + compute_offset(indices...));
        }
        //
        inline void set_live_range(int start, int end) {
            this->_start = start;
            this->_end = end;
        }
        inline int start() const { return _start; }
        inline int end() const { return _end; }
    private:
        template<typename... Args>
        size_t compute_offset(Args... indices) const {
            constexpr size_t num_indices = sizeof...(indices);
            if (num_indices > _shape.size()) {
                throw std::invalid_argument(
                    "Number of indices (" + std::to_string(num_indices) +
                    ") does not match tensor dimensions (" + std::to_string(_shape.size()) + ")"
                );
            }
            std::array<size_t, num_indices> idxs{static_cast<size_t>(indices)...};
            size_t offset = 0;
            for (size_t i = 0; i < num_indices; ++i) {
                offset += idxs[i] * _strides[i];
            }

            return offset;
        }

    private:
        //
        //
        int _priority = 0;
        bool _is_allocated;
        bool _use_external = false;
        size_t _external_memory_index = -1;
        size_t _n_dims{};
        tff::core::memory::DataType _data_type;
        tff::core::memory::ModelTensorType _tensor_type;
        size_t _type_size{};
        uint32_t _blk_size{};
        std::array<int64_t, MAX_TENSOR_DIM> _shape{1,1,1,1};
        std::array<int64_t, MAX_TENSOR_DIM> _strides{0,0,0,0};
        std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> _allocator;
        std::shared_ptr<tff::core::memory::Memory> _buffer;


        //
        //用于内存管理;
        int _start;  // 活跃开始时间点
        int _end;    // 活跃结束时间点


    private:
        mutable std::mutex _mutex;
    };

    //
}

#endif //TFFINFER_TENSOR_H
