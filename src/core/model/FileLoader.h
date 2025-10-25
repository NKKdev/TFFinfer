//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_FILELOADER_H
#define TFFINFER_FILELOADER_H
#include <memory>
#include "Logger.h"
#include "util.h"
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#include <fcntl.h>
#endif
//
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
//#include <bits/resource.h>
#endif
namespace tff::core::model {
    //file
    class FileLoader {
    public:
        FileLoader(const char *fname, const char *mode) {
            _fp = fopen(fname, mode);
            if (_fp == nullptr) {
                tff::log::Logger::error("failed to open %s: %s", fname, strerror(errno));
                return;
            }
            seek(0, SEEK_END);
            _file_size = tell();
            seek(0, SEEK_SET);
        };

        ~FileLoader() {
            if (_fp != nullptr) {
                fclose(_fp);_fp = nullptr;
            }
        };

    public:
        //
        bool file_aligned(const size_t &alignment) const;
        //
        [[nodiscard]] size_t tell() const;

        [[nodiscard]] size_t size() const;

        [[nodiscard]] int file_id() const;

        void seek(size_t offset, int whence) const;

        void read_raw(void *ptr, const size_t &len) const;

        [[nodiscard]] uint32_t read_u32() const;

        void write_raw(const void *ptr, const size_t &len) const;

        void write_u32(const uint32_t &val) const;

        template<typename T>
        inline bool read(T &dst) const {
            return fread(&dst, 1, sizeof(dst), _fp) == sizeof(dst);
        }

        template<typename T>
        inline bool read(std::vector<T> &dst, const size_t n) const {
            dst.resize(n);
            for (size_t i = 0; i < dst.size(); ++i) {
                if constexpr (std::is_same<T, bool>::value) {
                    bool tmp;
                    if (!read(tmp)) {
                        return false;
                    }
                    dst[i] = tmp;
                } else {
                    if (!read(dst[i])) {
                        return false;
                    }
                }
            }
            return true;
        }

        inline bool read(bool &dst) const {
            int8_t tmp = -1;
            if (!read(tmp)) {
                return false;
            }
            dst = tmp != 0;
            return true;
        }

    private:
        FILE *_fp;
        size_t _file_size;
    };

    inline bool FileLoader::file_aligned(const size_t &alignment) const {
        if (fseek(this->_fp, TFF_PAD(ftell(this->_fp), alignment), SEEK_SET) != 0) {
            return false;
        }
        return true;
    }

    inline size_t FileLoader::tell() const {
        const size_t ret = std::ftell(_fp);
        if (ret == -1) {
            tff::log::Logger::error("ftell error: %s", strerror(errno));
        }
        return ret;
    }

    inline size_t FileLoader::size() const {
        return _file_size;
    }

    inline int FileLoader::file_id() const {
        return ::fileno(_fp);
    }

    inline void FileLoader::seek(size_t offset, int whence) const {
        const int ret = std::fseek(_fp, (long) offset, whence);
        if (ret != 0) {
            tff::log::Logger::error("seek error: %s", strerror(errno));
        }
    }

    inline void FileLoader::read_raw(void *ptr, const size_t &len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, _fp);
        if (ferror(_fp)) {
            tff::log::Logger::error("read error: %s", strerror(errno));
        }
        if (ret != 1) {
            tff::log::Logger::error("unexpectedly reached end of file");
        }
    }

    inline uint32_t FileLoader::read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    inline void FileLoader::write_raw(const void *ptr, const size_t &len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        const size_t ret = std::fwrite(ptr, len, 1, _fp);
        if (ret != 1) {
            tff::log::Logger::error("write error: %s", strerror(errno));
        }
    }

    inline void FileLoader::write_u32(const uint32_t &val) const {
        write_raw(&val, sizeof(val));
    }

    //map
    class FileMMap {
    public:
        FileMMap(const FileMMap &) = delete;

        explicit FileMMap(const std::shared_ptr<FileLoader> &file_loader,
                          size_t prefetch = static_cast<size_t>(-1), bool numa = false) {
            _size = file_loader->size();
            const int fd = file_loader->file_id();
            int flags = MAP_SHARED;
            if (numa) { prefetch = 0; }

            if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
                tff::log::Logger::warning("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                                          strerror(errno));
            }
            if (prefetch) { flags |= MAP_POPULATE; }

            _addr = mmap(NULL, file_loader->size(), PROT_READ, flags, fd, 0);
            if (_addr == MAP_FAILED) {
                tff::log::Logger::error("mmap failed: %s", strerror(errno));
            }

            if (prefetch > 0) {
                if (posix_madvise(_addr, std::min(file_loader->size(), prefetch), POSIX_MADV_WILLNEED)) {
                    tff::log::Logger::warning("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                                              strerror(errno));
                }
            }
            if (numa) {
                if (posix_madvise(_addr, file_loader->size(), POSIX_MADV_RANDOM)) {
                    tff::log::Logger::warning("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                                              strerror(errno));
                }
            }

            _mapped_fragments.emplace_back(0, file_loader->size());
        }

        ~FileMMap() = default;

        [[nodiscard]] size_t size() const;

        [[nodiscard]] void *addr() const;

        void unmap_fragment(size_t &first, size_t &last);

        //
        void align_range(size_t *first, size_t *last, size_t page_size);

    private:
        std::vector<std::pair<size_t, size_t> > _mapped_fragments;
        void *_addr;
        size_t _size;
    };

    inline size_t FileMMap::size() const {
        return _size;
    }

    inline void *FileMMap::addr() const {
        return _addr;
    }

    inline void align_range(size_t *first, size_t *last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    inline void FileMMap::unmap_fragment(size_t &first, size_t &last) {
        const int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        const size_t len = last - first;

        if (len == 0) {
            return;
        }

        void *next_page_start = static_cast<uint8_t *>(_addr) + first;

        if (munmap(next_page_start, len)) {
            tff::log::Logger::warning("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t> > new_mapped_fragments;
        for (const auto &frag: _mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        _mapped_fragments = std::move(new_mapped_fragments);
    }

    //file lock
    class FileLock {
    public:
        FileLock() = default;

        ~FileLock() = default;

    public:
        void init(void *ptr);

        void grow_to(size_t target_size);

        bool raw_lock(const void *addr, size_t size) const;

        void raw_unlock(void *addr, size_t size) const;

    private:
        void *_addr;
        size_t _size;

        bool _failed_already;
    };

    inline void FileLock::init(void *ptr) {
        _addr = static_cast<uint8_t *>(ptr);
    }

    inline void FileLock::grow_to(size_t target_size) {
        if (_failed_already) {
            return;
        }
        size_t granularity = sysconf(_SC_PAGESIZE);
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > _size) {
            if (raw_lock((uint8_t *) _addr + _size, target_size - _size)) {
                _size = target_size;
            } else {
                _failed_already = true;
            }
        }
    }

    inline bool FileLock::raw_lock(const void *addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }


        char *errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);

        rlimit lock_limit{};
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }

        return false;
    }

    inline void FileLock::raw_unlock(void *addr, size_t size) const {
        if (munlock(addr, size)) {
            tff::log::Logger::warning("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
}


#endif //TFFINFER_FILELOADER_H
