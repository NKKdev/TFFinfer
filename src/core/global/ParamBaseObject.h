//
// Created by nkk on 2025/11/14.
//

#ifndef TFFINFER_PARAMBASEOBJECT_H
#define TFFINFER_PARAMBASEOBJECT_H
#include <memory>
#include <cstring>
#include <set>
#include <type_traits>
#include <stdexcept>
#include "global/GlobalDefine.h"
#include "log/Logger.h"

namespace tff::core::global {
    static constexpr size_t BUFFER_SIZE = MAX_PARAM_BUFFER_SIZE;
    static constexpr size_t MAX_PARAMS = MAX_PARAM_COUNT;
    /**
     * @brief 参数对象
     */
    class ParamBaseObject final : std::enable_shared_from_this<ParamBaseObject> {
    public:
        ParamBaseObject() : _use_para_count(0) {
        };

        virtual ~ParamBaseObject() = default;

        ParamBaseObject(const ParamBaseObject &) = default;

        ParamBaseObject(ParamBaseObject &&) = default;

        ParamBaseObject &operator=(const ParamBaseObject &other) {
            if (this == &other) {
                return *this;
            }
            _params = other._params;
            _use_para_count = other._use_para_count;
            return *this;
        }

        ParamBaseObject &operator=(ParamBaseObject &&other) noexcept {
            if (this == &other) {
                return *this;
            }
            _params = std::move(other._params);
            _use_para_count = other._use_para_count;
            other._use_para_count = 0;
            return *this;
        }

    public:
        /**
         * @brief 设置参数
         * @tparam T 参数类型
         * @param name 参数名称
         * @param value 参数值
         */
        template<typename T>
        void set_param(const std::string &name, T &&value) {
            _params[name] = std::forward<T>(value);
            _param_names.insert(name);
        }

        /**
         * @brief 获取参数
         * @tparam T 参数类型
         * @param name 参数名称
         * @return 参数值
         */
        template<typename T>
        std::optional<std::reference_wrapper<const T> > get_param(const std::string &name) const {
            const auto it = _params.find(name);
            if (it == _params.end()) {
                //tff::log::Logger::warning("Parameter '%s' not found", name.c_str());
                return std::nullopt;
            }
            try {
                const T &val = std::any_cast<const T &>(it->second);
                return std::cref(val);
            } catch (const std::bad_any_cast &e) {
                tff::log::Logger::error("Type mismatch for param '%s'", name.c_str());
                return std::nullopt;
            }
        }

        /**
         * @brief 获取参数数量
         * @return 参数数量
         */
        size_t get_param_count() const { return _param_names.size(); }
        /**
         * @brief 获取参数
         * @return 参数
         */
        const std::unordered_map<std::string, std::any> &params() const {
            return _params;
        }

    private:
        std::unordered_map<std::string, std::any> _params;
        std::set<std::string> _param_names;
        int _use_para_count;
    };
}


#endif //TFFINFER_PARAMBASEOBJECT_H
