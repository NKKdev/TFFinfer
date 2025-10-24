//
// Created by nkk on 2025/4/27.
//

#ifndef DEEP_TFF_MODULEFACTORY_H
#define DEEP_TFF_MODULEFACTORY_H
#include "ExportInc.h"
#include "ModuleObject.h"
#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
namespace tff::factory {
  class DEEP_TFF_API ModuleFactory {
  public:
    // 单例访问
    static std::shared_ptr<ModuleFactory> instance();

  public:
    using ptr_type = std::shared_ptr<tff::module::ModuleObject>;
    template <typename T> class FactoryRegister_t {

    public:
      FactoryRegister_t(const std::string &_module_type, const std::string &key) {
        ModuleFactory::instance()->m_module_func_map[_module_type].emplace(
          key, [] { return ptr_type(new T()); });
      }

      template <typename... Args>
      FactoryRegister_t(const std::string &_module_type, const std::string &key,
                        Args... args) {
        ModuleFactory::instance()->m_module_func_map[_module_type].emplace(
          key, [=] { return ptr_type(new T(args...)); });
      }
    };

    ptr_type create(const std::string &_module_type, const std::string &key);
    std::unique_ptr<tff::module::ModuleObject> create_unique(const std::string &_module_type,
                                                             const std::string &key);
    std::shared_ptr<tff::module::ModuleObject> create_shared(const std::string &_module_type,
                                                             const std::string &key);

  private:
    ModuleFactory() = default;
  public:
    ModuleFactory(const ModuleFactory &) = delete;
    ModuleFactory(ModuleFactory &&) = delete;

  public:
    // std::unordered_map<std::string, std::function<ptr_type()>> map_;
    std::unordered_map<std::string,
      std::unordered_map<std::string, std::function<ptr_type()>>>
    m_module_func_map;
    //
    std::unordered_map<std::string, std::unordered_map<std::string, ptr_type>>
    m_module_map;
  };
#define REGISTER_MODULE_OBJECT_NAME(T) reg_msg_##T##_
#define REGISTER_MODULE_OBJECT(T, type, key, ...)                              \
  static tff::factory::ModuleFactory::FactoryRegister_t<T> REGISTER_MODULE_OBJECT_NAME(T)(  \
      type, key, ##__VA_ARGS__);
}
#endif // DEEP_TFF_MODULEFACTORY_H
