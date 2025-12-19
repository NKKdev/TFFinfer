//
// Created by nkk on 2025/11/2.
//

#ifndef TFFINFER_LLMKVCACHE_H
#define TFFINFER_LLMKVCACHE_H
#include <memory>
#include <mutex>
#include "Tensor.h"
#include "global/OPDefine.h"
#include "global/GlobalDefine.h"
namespace tff::core::memory {
#define INVALID_PAGE_ID uint64_t(-1)
#define PAGE_SIZE 32
#define MAX_TOKENS
    using PageID = uint64_t;
    using DeviceTensor = tff::core::memory::Tensor;

    struct KVPage {
        std::unique_ptr<tff::core::memory::Tensor> _k; // [n_embd_head, n_head_kv, PAGE_SIZE]
        std::unique_ptr<tff::core::memory::Tensor> _v; // [n_embd_head, n_head_kv, PAGE_SIZE]

        int _n_tokens = 0; // 当前已缓存的 token 数（写入位置）
        bool _is_used = false;
        mutable std::mutex mutex; // 可选锁
    };

    class PageManager : public std::enable_shared_from_this<PageManager> {
    public:
        PageManager(const tff::core::memory::DataType data_type,
                    int total_pages, int _d_h, int _h_kv,
                    std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> allocator =
                            nullptr, int page_size = PAGE_SIZE)
            : _total_pages(total_pages), _page_size(page_size), _n_d_h(_d_h), _n_h_kv(_h_kv) {
            _pages.resize(total_pages);
            for (int i = 0; i < total_pages; ++i) {
                _pages[i] = std::make_unique<KVPage>();
                std::array<int64_t, MAX_TENSOR_DIM> shapes = {static_cast<int64_t>(_d_h), static_cast<int64_t>(_h_kv), static_cast<int64_t>(page_size), 1};
                _pages[i]->_k = std::make_unique<tff::core::memory::Tensor>(3, data_type, shapes, false, allocator);
                _pages[i]->_v = std::make_unique<tff::core::memory::Tensor>(3, data_type, shapes, false, allocator);
                _pages[i]->_n_tokens = 0;
                _pages[i]->_is_used = false;
                _free_list.push_back(i);
            }
        };

        ~PageManager() {
            std::lock_guard<std::mutex> lock(_mutex);
            // for (int i = 0; i < _total_pages; ++i) {
            //     auto &page = _pages[i];
            //     if (page->_k) {
            //         page->_k->release();
            //     }
            //     if (page->_v) {
            //         page->_v->release();
            //     }
            //
            // }
            _free_list.clear();
            _pages.clear();
            _total_pages = 0;
        }

    public:
        // 分配一个新 page
        PageID allocate() {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_free_list.empty()) return INVALID_PAGE_ID;
            PageID id = _free_list.back();
            _free_list.pop_back();
            _pages[id]->_is_used = true;
            return id;
        }

        // 释放一个 page
        void free(PageID id) {
            std::lock_guard<std::mutex> lock(_mutex);
            if (id >= 0 && id < (int) _pages.size() && _pages[id]->_is_used) {
                _pages[id]->_is_used = false;
                _free_list.push_back(id);
            }
        }

        // 获取某个 page 的指针（用于 kernel 调用）
        const DeviceTensor *get_k(PageID id) const {
            return id != INVALID_PAGE_ID ? _pages[id]->_k.get() : nullptr;
        }

        const DeviceTensor *get_v(PageID id) const {
            return id != INVALID_PAGE_ID ? _pages[id]->_v.get() : nullptr;
        }

        int used_count() const {
            std::lock_guard<std::mutex> lock(_mutex);
            return (int) _pages.size() - (int) _free_list.size();
        }

        //
        inline int32_t get_page_size() const { return _page_size; }

    private:
        uint32_t _total_pages = 0;
        std::vector<std::unique_ptr<KVPage> > _pages;
        std::vector<PageID> _free_list; // 空闲 page 的 ID 列表
        int _page_size;
        int _n_d_h, _n_h_kv;
        mutable std::mutex _mutex;
    };

    //
    class LayerKVContext : public std::enable_shared_from_this<LayerKVContext> {
    public:
        LayerKVContext();

        LayerKVContext(const int sid, const int lid, const std::shared_ptr<PageManager> &page_manager) : _seq_id(sid), _layer_id(lid),
        _page_manager(page_manager){
        }

        ~LayerKVContext()= default;

    public:
        int get_num_pages() const { return (int) _page_table.size(); }
        int get_max_tokens() const { return _page_table.size() * _page_manager->get_page_size(); }
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
            int _page_size = 32; // 每个 page 存多少个 token
            bool _use_f16 = true;
        };

    public:
        explicit LLMKVCache(const tff::core::memory::DataType data_type, const LLMKVCache::KVConfig &cfg,
                            std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> allocator = nullptr)
            : _config(cfg),_seq_length(MAX_SEQ_LENGTH) {
            this->_page_manager = std::make_shared<PageManager>(
                data_type,
                cfg._total_pages,
                cfg._n_embd_head,
                cfg._n_head_kv,
                allocator,
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
        inline void begine_prefill(const size_t &batch_size,const size_t &seq_len) {
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

        bool set(int seq_id, int layer_id, const DeviceTensor *cur_k, const DeviceTensor *cur_v) {
            LayerKVContext *ctx = get_context(seq_id, layer_id);
            if (!ctx->append_token()) {
                return false; // 内存不足
            }

            int token_pos = ctx->_num_tokens - 1;

            auto index_pair = ctx->get_location(token_pos);
            const int page_idx = index_pair.first;
            const int offset   = index_pair.second;

            PageID page_id = ctx->_page_table[page_idx];
            //DeviceTensor* dst_k = slice_page(_page_manager->get_k(page_id), offset);
            //DeviceTensor* dst_v = slice_page(_page_manager->get_v(page_id), offset);

            // 执行拷贝：cur_k -> dst_k（形状 [d_h, h_kv, 1]）
            //copy_tensor(cur_k, dst_k);
            //copy_tensor(cur_v, dst_v);

            return true;
        }

        std::tuple<std::shared_ptr<tff::core::memory::Tensor>, std::shared_ptr<tff::core::memory::Tensor>, const PageID
            *, int>
        get(int seq_id, int layer_id, int from_token = 0) {
            const LayerKVContext *ctx = get_context(seq_id, layer_id);
            if (from_token >= ctx->_num_tokens) {
                return {nullptr, nullptr, nullptr, 0};
            }

            const PageID *block_table = ctx->_page_table.data();
            int num_blocks = ctx->get_num_pages();

            return {nullptr, nullptr, block_table, num_blocks};
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
