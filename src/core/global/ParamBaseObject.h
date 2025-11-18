//
// Created by nkk on 2025/11/14.
//

#ifndef TFFINFER_PARAMBASEOBJECT_H
#define TFFINFER_PARAMBASEOBJECT_H
#include <memory>
#include <cstring>
#include <type_traits>
#include <stdexcept>
#include "global/GlobalDefine.h"
#include "log/Logger.h"

namespace tff::core::global {
    static constexpr size_t BUFFER_SIZE = MAX_PARAM_BUFFER_SIZE;
    static constexpr size_t MAX_PARAMS = MAX_PARAM_COUNT;

    class ParamBaseObject final : std::enable_shared_from_this<ParamBaseObject> {
    public:
        ParamBaseObject() : _use_para_count(0) {
        };

        virtual ~ParamBaseObject() = default;

        ParamBaseObject(const ParamBaseObject &) = default;

        ParamBaseObject(ParamBaseObject &&) = default;

        ParamBaseObject& operator=(const ParamBaseObject& other) {

            if (this == &other) {
                return *this;
            }
            _params = other._params;
            _use_para_count = other._use_para_count;
            return *this;
        }
        ParamBaseObject& operator=(ParamBaseObject&& other) noexcept {
            if (this == &other) {
                return *this;
            }
            _params = std::move(other._params);
            _use_para_count = other._use_para_count;
            other._use_para_count = 0;
            return *this;
        }

    public:
        template<typename T>
        void set_param(size_t index, T &&value) {
            if (index >= MAX_PARAMS) {
                tff::log::Logger::error("Parameter index out of range");
                return;
            }
            _params[index] = std::forward<T>(value);
            if (index >= _use_para_count) {
                _use_para_count++;
            }
        }

        template<typename T>
        std::optional<std::reference_wrapper<const T> > get_param(size_t index) const {
            if (index >= MAX_PARAMS || !_params[index].has_value()) {
                tff::log::Logger::warning("get_param: index %zu out of range or empty", index);
                return std::nullopt;
            }
            try {
                const T &val = std::any_cast<const T &>(_params[index]);
                return std::cref(val);
            } catch (const std::bad_any_cast &) {
                tff::log::Logger::error("get_param: type mismatch at index %zu (requested %s)", index,
                                        typeid(T).name());
                return std::nullopt;
            }
        }

        //
        template<typename T>
        std::optional<std::reference_wrapper<T> > get_param_mut(size_t index) {
            if (index >= MAX_PARAMS || !_params[index].has_value()) {
                tff::log::Logger::warning("get_param_mut: index %zu out of range or empty", index);
                return std::nullopt;
            }
            try {
                T &val = std::any_cast<T &>(_params[index]);
                return std::ref(val);
            } catch (const std::bad_any_cast &) {
                tff::log::Logger::error("get_param_mut: type mismatch at index %zu (requested %s)", index,
                                        typeid(T).name());
                return std::nullopt;
            }
        }

        //
        inline size_t get_param_count() const {
            return _use_para_count;
        }

    private:
        std::array<std::any, MAX_PARAMS> _params;
        int32_t _use_para_count;
    };
}


#endif //TFFINFER_PARAMBASEOBJECT_H
