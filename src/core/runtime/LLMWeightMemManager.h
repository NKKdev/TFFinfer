//
// Created by nkk on 2025/11/16.
//

#ifndef TFFINFER_LLMWEIGHTMEMMANAGER_H
#define TFFINFER_LLMWEIGHTMEMMANAGER_H
#include <memory>
#include <mutex>
#include "mem/Memory.h"
#include "log/Logger.h"

namespace tff::core::runtime {
    class LLMWeightMemManager final : public tff::module::ModuleObject {
    public:
        LLMWeightMemManager() = default;

        ~LLMWeightMemManager() override {
            if (this->_cpu_mapped_buffer.size() > 0) {
                this->_cpu_mapped_buffer.clear();
            }
            if (this->_gpu_buffer.size() > 0) {

                this->_gpu_buffer.clear();
            }
            if (this->_pinned_buffer.size() > 0) {

                this->_pinned_buffer.clear();
            }
        };
    public:
        //
        bool init(const size_t &buffer_size);
        //
        inline std::pair<int, void*> get_cpu_mapped_memory() const {
            std::lock_guard<std::mutex> lock(_mutex);
            int index = 0;
            for (const auto &it : _cpu_mapped_buffer) {
                if ((it)->is_used()) {
                    index++;
                    continue;
                }
                it->occupy();
                return std::make_pair(index,it->ptr());
            }
            return std::make_pair(-1,nullptr);;
        }

        //
        inline std::pair<int, void*> get_pinned_memory() const {
            std::lock_guard<std::mutex> lock(_mutex);
            int index = 0;
            for (const auto &it : _pinned_buffer) {
                if ((it)->is_used()) {
                    index++;
                    continue;
                }
                it->occupy();
                return std::make_pair(index,it->ptr());
            }
            return std::make_pair(-1,nullptr);
        }
        //
        inline std::pair<int, void*> get_gpu_memory() const {
            std::lock_guard<std::mutex> lock(_mutex);
            int index = 0;
            for (const auto &it : _gpu_buffer) {
                if ((it)->is_used()) {
                    index++;
                    continue;
                }
                it->occupy();
                return std::make_pair(index,it->ptr());
            }
            return std::make_pair(-1,nullptr);
        }
        //
        inline void reset_cpu_mapped_memory(const size_t &index) {
            std::lock_guard<std::mutex> lock(_mutex);
            this->_cpu_mapped_buffer[index]->reset();
        }
        //
        inline void reset_gpu_memory(const size_t &index) {
            std::lock_guard<std::mutex> lock(_mutex);
            this->_gpu_buffer[index]->reset();
        }
        //
        inline void reset_pinned_memory(const size_t &index) {
            std::lock_guard<std::mutex> lock(_mutex);
            this->_pinned_buffer[index]->reset();
        }
    private:
        bool init_cpu_buffer(const size_t &buffer_size);
        bool init_gpu_buffer(const size_t &buffer_size);
    private:
        std::vector<std::shared_ptr<tff::core::memory::Memory>> _cpu_mapped_buffer;
        std::vector<std::shared_ptr<tff::core::memory::Memory>> _pinned_buffer;
        std::vector<std::shared_ptr<tff::core::memory::Memory>> _gpu_buffer;
        //
        mutable std::mutex _mutex;
    };
}

#endif //TFFINFER_LLMWEIGHTMEMMANAGER_H
