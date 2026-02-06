//
// Created by nkk on 2025/11/2.
//

#ifndef TFFINFER_LLMKVCACHE_H
#define TFFINFER_LLMKVCACHE_H
#include <memory>
#include <mutex>
#include "mem/Tensor.h"
#include "global/GlobalDefine.h"
#include "device/DeviceBaseObject.h"
#include "MemManager.h"

namespace tff::core::runtime {
#define INVALID_PAGE_ID uint64_t(-1)
#define PAGE_SIZE 32
#define MAX_TOKENS
    using PageID = uint64_t;
    using DeviceTensor = tff::core::memory::Tensor;

    struct KVPage {
        std::shared_ptr<tff::core::memory::Tensor> _k;
        std::shared_ptr<tff::core::memory::Tensor> _v;

        int _n_tokens = 0; // 当前已缓存的 token 数（写入位置）
        bool _is_used = false;
        mutable std::mutex mutex; // 可选锁
    };

    class PageManager : public std::enable_shared_from_this<PageManager> {
    public:
        PageManager(const int &device_id,
                    const tff::core::memory::DataType data_type,
                    int total_pages, int _d_h, int _h_kv,
                    std::shared_ptr<tff::core::runtime::LLMMemManager> &mem_manager_ptr,
                    int page_size = PAGE_SIZE)
            : _device_id(device_id), _total_pages(total_pages), _page_size(page_size), _n_d_h(_d_h), _n_h_kv(_h_kv),
              _mem_manager_ptr(mem_manager_ptr) {
            _pages.resize(total_pages);
            for (int i = 0; i < total_pages; ++i) {
                _pages[i] = std::make_shared<KVPage>();
                std::array<int64_t, MAX_TENSOR_DIM> shapes = {
                    static_cast<int64_t>(_d_h) , static_cast<int64_t>(_h_kv), static_cast<int64_t>(page_size), 1
                };
                _pages[i]->_k = std::make_shared<tff::core::memory::Tensor>(
                    data_type, memory::MemoryType::TFF_MEM_TYPE_KV_CACHE,
                    shapes);
                _pages[i]->_k->set_external_memory_index(mem_manager_ptr->allocate_memory_offset(
                    _pages[i]->_k->get_bytes(),
                    _device_id, memory::MemoryType::TFF_MEM_TYPE_KV_CACHE));
                _pages[i]->_v = std::make_shared<tff::core::memory::Tensor>(
                    data_type, memory::MemoryType::TFF_MEM_TYPE_KV_CACHE,
                    shapes);
                _pages[i]->_v->set_external_memory_index(mem_manager_ptr->allocate_memory_offset(
                    _pages[i]->_v->get_bytes(),
                    _device_id, memory::MemoryType::TFF_MEM_TYPE_KV_CACHE));
                _pages[i]->_n_tokens = 0;
                _pages[i]->_is_used = false;
                _free_list.push_back(i);
            }
        };

        ~PageManager() {
            std::lock_guard<std::mutex> lock(_mutex);
            _free_list.clear();
            _pages.clear();
            _total_pages = 0;
        }

    public:
        // 分配一个新 page
        inline PageID allocate() {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_free_list.empty()) return INVALID_PAGE_ID;
            PageID id = _free_list.back();
            _free_list.pop_back();
            _pages[id]->_is_used = true;
            return id;
        }

        // 释放一个 page
        inline void free(PageID id) {
            std::lock_guard<std::mutex> lock(_mutex);
            if (id >= 0 && id < (int) _pages.size() && _pages[id]->_is_used) {
                _pages[id]->_is_used = false;
                _free_list.push_back(id);
            }
        }

        // 获取某个 page 的指针（用于 kernel 调用）
        inline std::shared_ptr<tff::core::memory::Tensor> get_k(PageID id) const {
            std::lock_guard<std::mutex> lock(_mutex);
            if (id != INVALID_PAGE_ID && id < static_cast<int>(_pages.size()) && _pages[id]->_is_used) {
                if (_pages[id]->_k->get_buffer() == nullptr) {
                    const auto [fst, snd] = _mem_manager_ptr->allocate_memory(_pages[id]->_k->get_bytes(),
                                                                          this->_device_id,
                                                                          memory::MemoryType::TFF_MEM_TYPE_KV_CACHE);
                    _pages[id]->_k->set_buffer_data(snd, _pages[id]->_k->get_bytes(),
                                                    fst);
                }

                return _pages[id]->_k;
            } else {
                return nullptr;
            }
        }

        inline std::shared_ptr<tff::core::memory::Tensor> get_v(PageID id) const {
            std::lock_guard<std::mutex> lock(_mutex);
            if (id != INVALID_PAGE_ID && id < static_cast<int>(_pages.size()) && _pages[id]->_is_used) {
                if (_pages[id]->_v->get_buffer() == nullptr) {
                    const auto [fst, snd] = _mem_manager_ptr->allocate_memory(_pages[id]->_v->get_bytes(),
                                                                          this->_device_id,
                                                                          memory::MemoryType::TFF_MEM_TYPE_KV_CACHE);
                    _pages[id]->_v->set_buffer_data(snd, _pages[id]->_v->get_bytes(),
                                                    fst);
                }

                return _pages[id]->_v;
            } else {
                return nullptr;
            }
        }

        int used_count() const {
            std::lock_guard<std::mutex> lock(_mutex);
            return (int) _pages.size() - (int) _free_list.size();
        }

        //
        inline int32_t get_page_size() const { return _page_size; }

    private:
        int _device_id;
        uint32_t _total_pages = 0;
        std::vector<std::shared_ptr<KVPage> > _pages;
        std::vector<PageID> _free_list; // 空闲 page 的 ID 列表
        int _page_size;
        int _n_d_h, _n_h_kv;
        std::shared_ptr<tff::core::runtime::LLMMemManager> _mem_manager_ptr;
        mutable std::mutex _mutex;
    };

    //
    class LayerKVContext : public std::enable_shared_from_this<LayerKVContext> {
    public:
        LayerKVContext();

        LayerKVContext(const int sid, const int lid, const std::shared_ptr<PageManager> &page_manager) : _seq_id(sid),
            _layer_id(lid),
            _page_manager(page_manager) {
        }

        ~LayerKVContext() = default;

    public:
        inline int get_num_pages() const { return static_cast<int>(_page_table.size()); }
        inline int get_max_tokens() const { return _page_table.size() * _page_manager->get_page_size(); }
        inline int get_token_count() const { return _num_tokens; }
        //
        // 获取第 idx 个 token 所在的 page_id 和 page 内偏移
        inline std::pair<PageID, int> get_location(int token_idx) const {
            if (token_idx >= _num_tokens) return {INVALID_PAGE_ID, 0};
            int page_id = token_idx / _page_manager->get_page_size();
            int offset = token_idx % _page_manager->get_page_size();
            return {_page_table[page_id], offset};
        }

        // 添加一个新 token（返回是否成功）
        inline bool append_token() {
            if (_num_tokens == get_max_tokens()) {
                PageID new_page = _page_manager->allocate();
                if (new_page == INVALID_PAGE_ID) return false;
                _page_table.push_back(new_page);
            }
            _num_tokens++;
            return true;
        }

        inline void clear() {
            for (PageID pid: _page_table) {
                if (pid != INVALID_PAGE_ID) {
                    _page_manager->free(pid);
                }
            }
            _page_table.clear();
            _num_tokens = 0;
        }

    public:
        int _seq_id;
        int _layer_id;
        std::vector<PageID> _page_table;
        int _num_tokens = 0;
        //
        std::shared_ptr<PageManager> _page_manager;
    };

    class LLMKVCache : public std::enable_shared_from_this<LLMKVCache> {
    public:
        struct KVConfig {
            uint32_t _n_layer{}; // 层数
            uint32_t _n_head{}; // query heads 数量
            uint32_t _n_head_kv{}; // key/value heads 数量（支持 GQA）
            uint32_t _n_embd_head{}; // 每个 head 的维度（如 128）
            uint32_t _max_tokens{}; // 最大缓存 token 数（固定长度或滑动窗口）
            bool _use_sliding_window = false; // 是否启用滑动窗口
            bool _thread_safe = false; // 是否线程安全
            int _total_pages; // 全局 page 总数（显存限制）
            int _page_size = PAGE_SIZE; // 每个 page 存多少个 token
            bool _use_f16 = true;
            tff::core::memory::DataType _data_type;
        };

    public:
        explicit LLMKVCache(const int &device_id, const tff::core::memory::DataType data_type,
                            const LLMKVCache::KVConfig &cfg,
                            std::shared_ptr<tff::core::runtime::LLMMemManager> &mem_manager_ptr)
            : _device_id(device_id), _config(cfg), _seq_length(MAX_SEQ_LENGTH) {
            this->_page_manager = std::make_shared<PageManager>(
                device_id,
                data_type,
                cfg._total_pages,
                cfg._n_embd_head,
                cfg._n_head_kv,
                mem_manager_ptr,
                cfg._page_size
            );
        }

        LLMKVCache(const LLMKVCache &) = delete;

        LLMKVCache(LLMKVCache &&) = delete;

        LLMKVCache &operator=(const LLMKVCache &) = delete;

        LLMKVCache &operator=(LLMKVCache &&) = delete;

        ~LLMKVCache() {
        };

    public:
        //
        void build_layer_kvcache_context(const int &seq_id, const int &layer_id);

        //
        inline void begine_prefill(const size_t &batch_size, const size_t &seq_len) {
            this->_seq_length = seq_len;
            this->_current_batch_size = batch_size;
            _is_prefilling = true;
        }

        //
        inline void end_prefill() {
            _is_prefilling = false;
        }

        static inline int make_key(const int seq_id, const int layer_id) {
            return (seq_id << 16) | layer_id;
        }


        LayerKVContext *get_context(int seq_id, int layer_id) {
            const int key = make_key(seq_id, layer_id);
            std::lock_guard<std::mutex> lock(global_mutex);

            auto it = this->_seq_contexts.find(key);
            if (it != this->_seq_contexts.end()) {
                return it->second.get();
            }

            auto ctx = std::make_unique<LayerKVContext>(seq_id, layer_id, this->_page_manager);
            LayerKVContext *ptr = ctx.get();
            this->_seq_contexts[key] = std::move(ctx);
            return ptr;
        }

        bool set_kv(int seq_id, int layer_id,
                   const int &token_num);
        //
        inline int get_kv_token_num(int seq_id, int layer_id) {
            LayerKVContext *ctx = get_context(seq_id, layer_id);
            if (ctx == nullptr) {
                return -1;
            }
            return ctx->get_token_count();
        }
        //
        inline std::pair<PageID, int> get_location(int seq_id, int layer_id, const int token_idx) {
            return get_context(seq_id, layer_id)->get_location(token_idx);
        }
        //
        inline std::shared_ptr<memory::Tensor> get_k(int seq_id, int layer_id, PageID page_id) {
            const LayerKVContext *ctx = get_context(seq_id, layer_id);
            if (ctx == nullptr) {
                tff::log::Logger::error("current layer (%d) kv cache context is invalid!!", layer_id);
                return std::shared_ptr<core::memory::Tensor>();
            }
            return ctx->_page_manager->get_k(page_id);
        }

        inline std::shared_ptr<memory::Tensor> get_v(int seq_id, int layer_id, PageID page_id) {
            const LayerKVContext *ctx = get_context(seq_id, layer_id);
            if (ctx == nullptr) {
                tff::log::Logger::error("current layer (%d) kv cache context is invalid!!", layer_id);
                return std::shared_ptr<core::memory::Tensor>();
            }
            return ctx->_page_manager->get_v(page_id);
        }

        void clear(int seq_id) {
            std::lock_guard<std::mutex> lock(global_mutex);
            for (int lid = 0; lid < _config._n_layer; ++lid) {
                int key = make_key(seq_id, lid);
                auto it = _seq_contexts.find(key);
                if (it != _seq_contexts.end()) {
                    it->second->clear();
                    _seq_contexts.erase(it);
                }
            }
        }

        const LLMKVCache::KVConfig &get_config() const { return _config; }

    public:
        int _device_id;
        size_t _seq_length;
        size_t _current_batch_size;
        bool _is_prefilling;
        KVConfig _config;
        //
        std::shared_ptr<PageManager> _page_manager;
        std::unordered_map<int, std::unique_ptr<LayerKVContext> > _seq_contexts;
        mutable std::mutex global_mutex;
    };
}
#endif //TFFINFER_LLMKVCACHE_H
