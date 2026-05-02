//
// Created by nkk on 2026/2/12.
//

#ifndef TFFINFER_BUILDER_H
#define TFFINFER_BUILDER_H
#include <nlohmann/thirdparty/hedley/hedley.hpp>

#include "global/ParamBaseObject.h"

namespace tff::core::device {
    class DeviceEvent;
    class DeviceStream;
}

namespace tff::core::runtime {
    class LLMMemManager;
}

namespace tff::kernel::builder {
    /**
     * @brief Parameter builder interface
     */
    class IParamBuilder {
    public:
        struct CommonParams {
            static constexpr const char *MemManager = "mem_manager";
            static constexpr const char *Event = "event";
            static constexpr const char *WaitList = "event_list";
            static constexpr const char *Stream = "stream";
            static constexpr const char *Name = "name";
            static constexpr const char *InputList = "input_list";
            static constexpr const char *Out = "out";
        };
    public:
        static void with(const std::shared_ptr<IParamBuilder> &builder1, const std::shared_ptr<IParamBuilder> &builder2) {
            auto params = builder1->extract_params();
            params.erase(CommonParams::Out);
            for (const auto& [key, value] : params) {
                builder2->build()->set_param(key, value);
            }
        }
        static void with_out(const std::shared_ptr<IParamBuilder> &builder1, const std::shared_ptr<IParamBuilder> &builder2) {
            auto params = builder1->extract_params();
            for (const auto& [key, value] : params) {
                builder2->build()->set_param(key, value);
            }
        }
    public:
        virtual ~IParamBuilder() = default;

        virtual std::shared_ptr<core::global::ParamBaseObject> build() = 0;

        virtual void set_mem_manager(std::any mem_mgr) = 0;

        virtual void set_stream(std::any stream) = 0;

        virtual void set_event(std::any event) = 0;

        virtual void set_wait_list(std::any event_list) = 0;

        virtual void set_name(std::any name) = 0;

        virtual void set_input_list(std::any input_list) = 0;

        virtual std::unordered_map<std::string, std::any> extract_params() = 0;
    };
    /**
     * @brief Parameter builder base class
     */
    template<typename Derived>
    class OpParamBuilderBase : public IParamBuilder {
    public:
        OpParamBuilderBase() {
            this->_params = std::make_shared<tff::core::global::ParamBaseObject>();
        }

    public:
        template<typename SubBuilder>
        OpParamBuilderBase &with(SubBuilder &builder) {
            static_assert(
                std::is_base_of_v<OpParamBuilderBase<SubBuilder>, SubBuilder>,
                "SubBuilder must inherit from OpParamBuilderBase"
            );

            auto params = builder.extract_params();
            return inject_prefixed_params(params);
        }

    public:
        [[nodiscard]] std::unordered_map<std::string, std::any> extract_params() override {
            return _params->params();
        }

        Derived &inject_prefixed_params(const std::unordered_map<std::string, std::any> &params) {
            for (const auto &[name, value]: params) {
                _params->set_param(name, value);
            }
            return static_cast<Derived &>(*this);
        }

        template<typename T>
        Derived &mem_manager(T &&value) {
            _params->set_param(CommonParams::MemManager, std::forward<T>(value));
            return static_cast<Derived &>(*this);
        }

        template<typename T>
        Derived &stream(T &&value) {
            _params->set_param(CommonParams::Stream, std::forward<T>(value));
            return static_cast<Derived &>(*this);
        }

        template<typename T>
        Derived &event(T &&value) {
            _params->set_param(CommonParams::Event, std::forward<T>(value));
            return static_cast<Derived &>(*this);
        }

        template<typename T>
        Derived &wait_list(T &&value) {
            _params->set_param(CommonParams::WaitList, std::forward<T>(value));
            return static_cast<Derived &>(*this);
        }

        void set_mem_manager(std::any mem_mgr) override {
            _params->set_param(CommonParams::MemManager, std::move(mem_mgr));
        }

        template<typename T>
        Derived &name(T &&value) {
            _params->set_param(CommonParams::Name, std::forward<T>(value));
            return static_cast<Derived &>(*this);
        }

        template<typename T>
        Derived &intput_list(T &&value) {
            _params->set_param(CommonParams::InputList, std::forward<T>(value));
            return static_cast<Derived &>(*this);
        }

        template<typename T>
        Derived &out(T &&value) {
            _params->set_param(CommonParams::Out, std::forward<T>(value));
            return static_cast<Derived &>(*this);
        }

        void set_stream(std::any stream) override {
            _params->set_param(CommonParams::Stream, std::move(stream));
        }

        void set_event(std::any event) override {
            _params->set_param(CommonParams::Event, std::move(event));
        }

        void set_wait_list(std::any event_list) override {
            _params->set_param(CommonParams::WaitList, std::move(event_list));
        }

        void set_name(std::any name) override {
            _params->set_param(CommonParams::Name, std::move(name));
        }

        void set_input_list(std::any input_list) override {
            _params->set_param(CommonParams::InputList, std::move(input_list));
        }

        std::shared_ptr<tff::core::global::ParamBaseObject> build() {
            return _params;
        }

        template<typename T>
        T out() {
            return get<T>(CommonParams::Out);
        }

    protected:
        template<typename T>
        Derived &set(const char *name, T &&value) {
            _params->set_param(name, std::forward<T>(value));
            return static_cast<Derived &>(*this);
        }

        template<typename T>
        T get(const char *key) const {
            auto it = _params->get_param<T>(key);
            if (!it.has_value()) {
                //tff::log::Logger::info("Parameter not set: %s", key);
                return T();
            }
            return it.value();
        }

    protected:
        std::shared_ptr<tff::core::global::ParamBaseObject> _params;
    };
}

#endif //TFFINFER_BUILDER_H
