//
// Created by nkk on 2025/7/1.
//

#ifndef DEEP_TFF_FUNCTIONFACTORY_H
#define DEEP_TFF_FUNCTIONFACTORY_H
#include "ExportInc.h"
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <any>
#include <utility>
//#include "log/Logger.h"
namespace tff {
namespace factory {

class DEEP_TFF_API FunctionFactory {
public:
  // 单例访问
  static std::shared_ptr<FunctionFactory> instance();

public:
  // 注册一个函数
  template <typename R, typename... Args>
  void register_callback(const std::string &key,
                         UECallback<R, Args...> callback) {
    m_functions[std::type_index(typeid(UECallback<R, Args...>))][key] =
        std::any(callback);
  }

  // 调用一个已注册的函数
  template <typename R, typename... Args>
  R invoke_callback(const std::string &key, Args... args) {
    auto it = m_functions.find(std::type_index(typeid(UECallback<R, Args...>)));
    if (it != m_functions.end()) {
      auto func_it = it->second.find(key);
      if (func_it != it->second.end()) {
        auto &func_any = func_it->second;
        return std::any_cast<UECallback<R, Args...>>(func_any)(args...);
      }
    }
    //Logger::error("Function not found for key: %s",key.c_str());
  }


  // 构造函数、拷贝构造函数和移动构造函数
  FunctionFactory() = default;

private:
  std::unordered_map<std::type_index,
                     std::unordered_map<std::string, std::any>>
      m_functions;

  FunctionFactory(const FunctionFactory &) = delete;
  FunctionFactory(FunctionFactory &&) = delete;

public:

};
} // namespace Factory
} // namespace TFF
#endif // DEEP_TFF_FUNCTIONFACTORY_H
