//
// Created by nkk on 2025/4/27.
//

#ifndef DEEP_TFF_MODULEOBJECT_H
#define DEEP_TFF_MODULEOBJECT_H
#include "ExportInc.h"
#include <string>
namespace tff {
namespace module {
class DEEP_TFF_API ModuleObject {
public:
  ModuleObject() = default;
  virtual ~ModuleObject() = default;

  ModuleObject(const ModuleObject &) = delete;

  ModuleObject &operator=(const ModuleObject &) = delete;

public:
  std::string m_module_name;
};
} // namespace Module
} // namespace TFF
#endif // DEEP_TFF_MODULEOBJECT_H
