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
        ParamBaseObject():_use_para_count(0){};
        virtual ~ParamBaseObject() = default;
        ParamBaseObject(const ParamBaseObject &) = default;
        ParamBaseObject(ParamBaseObject &&) = default;
        ParamBaseObject &operator=(const ParamBaseObject &) = default;
        ParamBaseObject &operator=(ParamBaseObject &&) = default;
    public:
        template<typename T>
        void set_param(size_t index, T&& value) {
            if (index >= MAX_PARAMS) {
                tff::log::Logger::error("Parameter index out of range");
                return;
            }
            _params[index] = std::forward<T>(value);
            _use_para_count++;
        }
        //
        template<typename T>
        const T &get_param(size_t index) const {
            if (index >= MAX_PARAMS) {
                tff::log::Logger::error("Parameter index out of range");
                return nullptr;
            }
            try {
                return std::any_cast<const T&>(_params[index]);
            } catch (const std::bad_any_cast&) {
               tff::log::Logger::error("Type mismatch in get_param: requested type does not match stored type");
            }
            return nullptr;
        }
        //
        template<typename T>
        T &get_param_mut(size_t index) {
            if (index >= MAX_PARAMS) {
                tff::log::Logger::error("Parameter index out of range");
                return nullptr;
            }
            try {
                return std::any_cast<T&>(_params[index]);
            } catch (const std::bad_any_cast&) {
                tff::log::Logger::error("Type mismatch in get_param_mut");
            }
            return nullptr;
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
