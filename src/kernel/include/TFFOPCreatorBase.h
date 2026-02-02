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
    static T get_param_value(const int &index, std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = para_ptr->get_param<std::string>(0);
        auto opt = para_ptr->get_param<T>(index);
        if (!opt.has_value()) {
            tff::log::Logger::error("op(%s) Failed to get param[%d]", std::string(name.value()).c_str(), index);
            return T{};
        }
        return opt.value();
    }
    static std::function<OP_CALLBACK_TYPE> get_op_func(std::shared_ptr<core::device::DeviceBaseObject> &device,
        const tff::core::graph::TffOpType &op_type, const tff::core::memory::DataType &data_type) {
        if (device == nullptr) {
            tff::log::Logger::error("op device is invalid!!");
            return nullptr;
        }
        auto it = core::global::TFF_OP_TYPE_MAP.find(op_type);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return nullptr;
        }
        std::string device_tag_name;
        switch (device->device_type()) {
            case core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU:
                device_tag_name = std::string(core::device::CUDATag::name);
                break;
                case core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU:
                device_tag_name = std::string(core::device::CPUTag::name());
                break;
            case core::device::TFF_BACKEND_DEVICE_TYPE_UNKNOWN:
                break;
        }
        std::string op_name = std::string(it->second) + std::string("_") + device_tag_name;
        switch (data_type) {
            case tff::core::memory::DataType::TFF_DATA_TYPE_F32:
                op_name = op_name + tff::core::global::get_type_suffix<float>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F64:
                op_name = op_name + tff::core::global::get_type_suffix<double>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I32:
                op_name = op_name + tff::core::global::get_type_suffix<int32_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I64:
                op_name = op_name + tff::core::global::get_type_suffix<int64_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I8:
                op_name = op_name + tff::core::global::get_type_suffix<uint8_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F16:
                op_name = op_name + tff::core::global::get_type_suffix<half>();
                break;
            case tff::core::memory::TFF_DATA_TYPE_Q8_0:
                op_name = op_name + tff::core::global::get_type_suffix<Q8_0>();
                break;
            case core::memory::TFF_DATA_TYPE_Q8_0_ALIGNED:
                op_name += tff::core::global::get_type_suffix<Q8_0_ALIGNED>();
            default:
                break;
        }


        return tff::factory::FunctionFactory::instance()->get_callback<tff::kernel::base::OP_CALLBACK_TYPE>(
            OP_NODE_FLAG,
            op_name);
    }

    // 算子基类模板
    template<typename Derived, typename T, typename DeviceTag>
    class OPCreatorBase {
    public:
        using value_type = T;
        using device_tag = DeviceTag;
        static std::string get_device_tag_name() {
            if constexpr (std::is_same_v<decltype(DeviceTag::name), const char*>) {
                return DeviceTag::name;
            } else if constexpr (std::is_invocable_v<decltype(&DeviceTag::name)>) {
                return DeviceTag::name();
            }
        }

        static std::string get_op_name() {
            auto it = core::global::TFF_OP_TYPE_MAP.find(Derived::op_type());
            if (it == core::global::TFF_OP_TYPE_MAP.end()) {
                tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
                return "";
            }
            std::string name = std::string(it->second);
            name += "_" + get_device_tag_name();
            name += tff::core::global::get_type_suffix<T>();
            return name;
        }

        static void op_before_hook(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
            auto tensor = get_param_value<std::shared_ptr<core::memory::Tensor>>(
                para_ptr->get_param_count() - 7, para_ptr);
            const auto &is_prefill = get_param_value<bool>(para_ptr->get_param_count() - 6, para_ptr);
            const auto &name = get_param_value<std::string>(para_ptr->get_param_count() - 5, para_ptr);
            tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), Derived::get_op_name().c_str());
            auto mem_buffer_manager_ptr = get_param_value<
                std::shared_ptr<
                    tff::core::runtime::LLMMemManager> >(para_ptr->get_param_count() - 4, para_ptr);
            auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent> >(
                para_ptr->get_param_count() - 3, para_ptr);
            auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent> > >(
                para_ptr->get_param_count() - 2, para_ptr);
            auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                para_ptr->get_param_count() - 1, para_ptr);
            if (stream == nullptr || event == nullptr || mem_buffer_manager_ptr == nullptr) {
                tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
                return;
            }
            for (auto &wait_event: event_list) {
                stream->wait_event(wait_event->get_native_event());
            }
        }

        static void op_after_hook(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
            auto input_tensors = get_param_value<std::vector<std::shared_ptr<core::memory::Tensor>>>(
               para_ptr->get_param_count() - 8, para_ptr);
            auto tensor = get_param_value<std::shared_ptr<core::memory::Tensor>>(
               para_ptr->get_param_count() - 7, para_ptr);
            const auto &is_prefill = get_param_value<bool>(para_ptr->get_param_count() - 6, para_ptr);
            const auto &name = get_param_value<std::string>(para_ptr->get_param_count() - 5, para_ptr);
            tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), Derived::get_op_name().c_str());
            auto mem_buffer_manager_ptr = get_param_value<
                std::shared_ptr<
                    tff::core::runtime::LLMMemManager> >(para_ptr->get_param_count() - 4, para_ptr);
            auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent> >(
                para_ptr->get_param_count() - 3, para_ptr);
            auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent> > >(
                para_ptr->get_param_count() - 2, para_ptr);
            auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                para_ptr->get_param_count() - 1, para_ptr);
            if (stream == nullptr || event == nullptr || mem_buffer_manager_ptr == nullptr) {
                tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
                return;
            }
            event->record(stream);
            if (is_prefill) {
                auto &device_id = tensor->get_allocator()->_device_id;
                for (auto &input: input_tensors) {
                    mem_buffer_manager_ptr->release_memory(device_id, input->get_external_memory_index());
                }
            }
        }

        // 注册函数
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


        // 编译时检查
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
#endif //TFFINFER_TFFOPCREATORBASE_H
