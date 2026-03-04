//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_TFFOPCREATORBASE_H
#define TFFINFER_TFFOPCREATORBASE_H
#include "FunctionFactory.h"
#include "core/global/GlobalDefine.h"
#include "global/ModelGlobalVar.h"
#include "global/ParamBaseObject.h"
#include "core/device/DeviceBaseObject.h"
#include "runtime/MemManager.h"

namespace tff::kernel::base {
    using OP_CALLBACK_TYPE = void(std::shared_ptr<tff::core::global::ParamBaseObject> &);

    template<typename T>
    T get_param_value(const char *name, std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto opt = para_ptr->get_param<T>(name);
        if (!opt.has_value()) {
            tff::log::Logger::error("op %s Failed to get param[%d]", name);
            return T{};
        }
        return opt.value();
    }

    std::function<OP_CALLBACK_TYPE> get_op_func(std::shared_ptr<core::device::MemBufferAllocatorBaseObject> &allocator,
                                                       const tff::core::graph::TffOpType &op_type,
                                                       const tff::core::memory::DataType &data_type);

    void op_before_hook(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

    void op_after_hook(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);
    /**
     * @brief 创建OP对象
     * @tparam Derived 继承的OP子类
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename Derived, typename T, typename DeviceTag>
    class OPCreatorBase {
    public:
        using value_type = T;
        using device_tag = DeviceTag;
        /**
         * @brief 获取设备类型名称
         * @return 设备类型名称
         */
        inline static std::string get_device_tag_name() {
            if constexpr (std::is_same_v<decltype(DeviceTag::name), const char *>) {
                return DeviceTag::name;
            } else if constexpr (std::is_invocable_v<decltype(&DeviceTag::name)>) {
                return DeviceTag::name();
            } else {
                return "";
            }
        }
        /**
         * @brief 获取OP名称
         * @return OP名称
         */
        inline static std::string get_op_name() {
            auto it = core::global::TFF_OP_TYPE_MAP.find(Derived::op_type());
            if (it == core::global::TFF_OP_TYPE_MAP.end()) {
                tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
                return "";
            }
            std::string name = std::string(it->second);
            name += "_" + get_device_tag_name();
            name += "_" + std::string(tff::core::global::get_type_suffix<T>());
            //tff::log::Logger::info("regist op (%s) ", name.c_str());
            return name;
        }
        /**
         * @brief 注册OP对象
         */
        static void registry_function() {
            static_assert(
                std::is_pointer_v<decltype(&Derived::compute)> &&
                std::is_function_v<std::remove_pointer_t<decltype(&Derived::compute)> >,
                "Derived must have static compute function"
            );
            static_assert(
                std::is_pointer_v<decltype(&Derived::op_type)> &&
                std::is_function_v<std::remove_pointer_t<decltype(&Derived::op_type)> >,
                "Derived must have static compute function"
            );
            auto callback = &Derived::compute;
            auto wrapper = [callback](std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
                op_before_hook(para_ptr);
                callback(para_ptr);
                op_after_hook(para_ptr);
            };


            tff::factory::FunctionFactory::instance()->register_callback(
                OP_NODE_FLAG,
                get_op_name(),
                wrapper
            );
        }
        /**
         * @brief 判断是否支持指定数据类型
         * @tparam CheckT 数据类型
         * @tparam CheckTag 设备类型
         * @return 是否支持
         */
        template<typename CheckT = T, typename CheckTag = DeviceTag>
        static constexpr bool is_supported() {
            return DeviceTag::template supports_data_type<CheckT>();
        }
    };
}
#define REGISTER_OP_OBJECT(T, type) \
namespace { \
    struct reg_##T##_##type##_##__LINE__ { \
        reg_##T##_##type##_##__LINE__() { \
            ::tff::kernel::T<type>::registry_function(); \
        } \
    } reg_##T##_##type##_##__LINE__##_instance; \
}

#define REGISTER_OP_OBJECT_DEVICE(T, type, tag) \
namespace { \
    struct reg_##T##_##type##_##__LINE__ { \
        reg_##T##_##type##_##__LINE__() { \
            ::tff::kernel::T<type, tag>::registry_function(); \
        } \
    } reg_##T##_##type##_##__LINE__##_instance; \
}
#endif //TFFINFER_TFFOPCREATORBASE_H
