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
    /**
     * @brief Tensor
     */
    class Tensor : public std::enable_shared_from_this<Tensor> {
    public:
        struct TensorCompare {
            bool operator()(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) const {
                return a->get_priority() < b->get_priority();
            }
        };

    public:
        Tensor(const std::shared_ptr<Tensor> &other) noexcept {
            release();
            _use_external = other->_use_external;
            _external_memory_index = other->_external_memory_index;
            _data_type = other->_data_type;
            _memory_type = other->_memory_type;
            _tensor_type = other->_tensor_type;
            _type_size = other->_type_size;
            _blk_size = other->_blk_size;
            _n_dims = other->_n_dims;
            _shape = other->_shape;
            _strides = other->_strides;
            _allocator = other->_allocator;
        }

        Tensor(const tff::core::memory::DataType data_type = tff::core::memory::DataType::TFF_DATA_TYPE_UNKNOWN,
               const MemoryType memory_type = MemoryType::TFF_MEM_TYPE_WORKSPACE,
               std::array<int64_t, MAX_TENSOR_DIM> shapes = std::array<int64_t, MAX_TENSOR_DIM>(),
               bool use_external = true,
               std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> alloc =
                       nullptr) : _is_allocated(false), _use_external(use_external), _data_type(data_type),
                                  _shape(std::move(shapes)),
                                  _allocator(std::move(alloc)),
                                  _memory_type(memory_type) {
            if (_shape.empty()) {
                tff::log::Logger::error("tensor shape is invalid!!");
                return;
            }
            this->check();
            this->set_data_type(data_type);
            if (!use_external) {
                this->allocate();
            }
        }

        Tensor &operator=(const Tensor &other) {
            if (this == &other) {
                return *this;
            }

            _use_external = other._use_external;
            _external_memory_index = other._external_memory_index;
            _data_type = other._data_type;
            _memory_type = other._memory_type;
            _tensor_type = other._tensor_type;
            _type_size = other._type_size;
            _blk_size = other._blk_size;
            _n_dims = other._n_dims;
            _shape = other._shape;
            _strides = other._strides;
            _allocator = other._allocator;
            this->_buffer = other._buffer;

            return *this;
        }

        Tensor &operator=(Tensor &&other) noexcept {
            if (this == &other) {
                return *this;
            }
            release();

            _use_external = other._use_external;
            _external_memory_index = other._external_memory_index;
            _is_allocated = other._is_allocated;
            _data_type = other._data_type;
            _memory_type = other._memory_type;
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

        ~Tensor() {
        }

    public:
        /**
         * @brief 分配张量内存
         */
        inline void allocate() {
            if (!_use_external) {
                _buffer = std::make_shared<tff::core::memory::Memory>(
                    type_traits_auto[this->_data_type]._type_size * std::accumulate(
                        _shape.begin(), _shape.end(), 1, std::multiplies<int64_t>()), nullptr, _use_external,
                    _allocator);
                _buffer->allocate();
                _is_allocated = true;
            }
        }

        /**
         * @brief 获取张量数据行大小
         * @return size_t 张量数据行大小
         */
        [[nodiscard]] inline size_t get_row_size() const {
            return _type_size * this->_shape[0] / type_traits_auto[_data_type]._blck_size;
        }

        /**
         * @brief 获取张量数据大小
         * @return size_t 张量数据大小
         */
        [[nodiscard]] inline int64_t get_bytes() const {
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

        /**
         * @brief 设置张量维度
         * @return
         */
        inline void set_dims(const size_t &n_dims) {
            this->_n_dims = n_dims;
        }

        /**
         * @brief 获取张量维度
         * @return int 张量维度
         */
        inline int dims() const {
            return this->_n_dims;
        }

        /**
         * @brief 获取张量形状
         * @return std::array<int64_t, MAX_TENSOR_DIM> 张量形状
         */
        inline std::array<int64_t, MAX_TENSOR_DIM> &get_shape() {
            return _shape;
        }

        /**
         * @brief 获取张量形状大小
         * @return std::array<int64_t, MAX_TENSOR_DIM> 张量形状
         */
        inline std::array<int64_t, MAX_TENSOR_DIM> &get_strides() {
            return this->_strides;
        }

        /**
         * @brief 设置张量形状
         * @param n_dims 张量维度
         * @param index 张量维度索引
         */
        inline void set_shape(const size_t &n_dims, const size_t &index) {
            this->_shape[index] = n_dims;
        }

        /**
         * @brief 设置张量形状
         * @param shape 张量形状
         */
        inline void set_shape(const std::array<int64_t, MAX_TENSOR_DIM> &shape) {
            this->_shape = shape;
            this->set_dims(this->_shape.size());
            stride_infer();
        }

        /**
         * @brief 获取张量内存类型
         * @return MemoryType 张量内存类型
         */
        inline MemoryType memory_type() const {
            return this->_memory_type;
        }

        /**
         * @brief 设置张量内存类型
         * @param memory_type 内存类型
         */
        inline void set_memory_type(const MemoryType &memory_type) {
            this->_memory_type = memory_type;
        }

        /**
         * @brief 获取张量数据类型
         * @return DataType 张量数据类型
         */
        inline DataType get_data_type() const {
            return this->_data_type;
        }

        /**
         * @brief 设置张量数据类型
         * @param data_type 数据类型
         */
        inline void set_data_type(const DataType &data_type) {
            this->_data_type = data_type;
            _type_size = memory::type_traits_auto[tff::core::memory::DataType(data_type)]._type_size;
            _blk_size = memory::type_traits_auto[tff::core::memory::DataType(data_type)]._blck_size;
            stride_infer();
        }

        /**
         * @brief 张量Stride推导
         * @return  void
         */
        inline void stride_infer() {
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

        /**
         * @brief 设置张量数据
         * @param data 数据指针
         * @param buffer_size 数据大小
         * @param mem_buffer_index 内存索引
         */
        inline void set_buffer_data(void *data, const size_t &buffer_size, size_t mem_buffer_index = -1) {
            _use_external = true;
            if (mem_buffer_index != -1) {
                this->_external_memory_index = mem_buffer_index;
            }
            this->_buffer = std::make_shared<tff::core::memory::Memory>(buffer_size, data, _use_external);
        }

        /**
         * @brief 获取张量外部内存索引
         * @return size_t 张量外部内存索引
         */
        inline int64_t get_external_memory_index() const {
            return _external_memory_index;
        }

        /**
         * @brief 设置张量外部内存索引
         * @param mem_offset 外部内存索引
         */
        inline void set_external_memory_index(const size_t &mem_offset) {
            this->_external_memory_index = mem_offset;
        }

        /**
         * @brief 获取张量类型
         * @return ModelTensorType 张量类型
         */
        [[nodiscard]] inline tff::core::memory::ModelTensorType get_tensor_type() const {
            return this->_tensor_type;
        }

        /**
         * @brief 设置张量类型
         * @param tensor_type 张量类型
         */
        inline void set_tensor_type(const tff::core::memory::ModelTensorType &tensor_type) {
            this->_tensor_type = tensor_type;
        }

        /**
         * @brief 释放张量内存
         */
        inline void release() {
            if (_buffer) {
                _allocator->release(_buffer->ptr());
            }
            _type_size = 0;
            _blk_size = 0;
        }

        /**
         * @brief 获取张量分配器
         * @return std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> 张量分配器
         */
        inline std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> &get_allocator() {
            return _allocator;
        }

        /**
         * @brief 设置张量分配器
         * @param allocator 张量分配器
         */
        inline void set_allocator(const std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> &allocator) {
            _allocator = allocator;
        }

        /**
         * @brief 张量是否分配
         * @return  bool
         */
        [[nodiscard]] inline bool is_allocated() const {
            return _is_allocated;
        }

        /**
         * @brief 获取张量内存
         * @return std::shared_ptr<tff::core::memory::Memory> 张量内存
         */
        inline std::shared_ptr<tff::core::memory::Memory> &get_buffer() {
            return _buffer;
        }

        /**
         * @brief 获取张量优先级
         * @return
         */
        [[nodiscard]] inline int get_priority() const {
            return _priority;
        }

        /**
         * @brief 设置张量优先级
         * @param priority 优先级
         */
        inline void set_priority(const int &priority) {
            _priority = priority;
        }

        /**
         * @brief 获取张量指针引用次数
         * @return size_t 张量指针引用次数
         */
        inline int get_ref_count() const {
            try {
                tff::log::Logger::info("current ref count: %d ", shared_from_this().use_count());
                return shared_from_this().use_count();
            } catch (const std::bad_weak_ptr &e) {
                return -1;
            }
        }

        /**
         * @brief 获取张量元素
         * @tparam T 张量元素类型
         * @param indices 索引
         * @return T& 张量元素
         */
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

        /**
         * @brief 获取张量元素
         * @tparam T 张量元素类型
         * @param indices 索引
         * @return T& 张量元素
         */
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

        inline void set_live_range(int start, int end) {
            this->_start = start;
            this->_end = end;
        }

        inline int start() const { return _start; }

        inline int end() const { return _end; }

    private:
        template<typename... Args>
        inline size_t compute_offset(Args... indices) const {
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

        //
        inline void check() {
            this->_n_dims = _shape.size();
            for (int i = _shape.size() - 1; i >= 0; --i) {
                if (_shape[i] == 1) {
                    this->_n_dims--;
                } else {
                    break;
                }
            }
        }

    private:
        //
        //
        int _priority = 0;
        bool _is_allocated;
        bool _use_external = false;
        int64_t _external_memory_index = -1;
        size_t _n_dims{};
        tff::core::memory::DataType _data_type;
        core::memory::MemoryType _memory_type;
        tff::core::memory::ModelTensorType _tensor_type;
        size_t _type_size{};
        uint32_t _blk_size{};
        std::array<int64_t, MAX_TENSOR_DIM> _shape{1, 1, 1, 1};
        std::array<int64_t, MAX_TENSOR_DIM> _strides{0, 0, 0, 0};
        std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> _allocator;
        std::shared_ptr<tff::core::memory::Memory> _buffer;


        //
        //用于内存管理;
        int _start; // 活跃开始时间点
        int _end; // 活跃结束时间点


    private:
        mutable std::mutex _mutex;
    };

    //
}

#endif //TFFINFER_TENSOR_H
