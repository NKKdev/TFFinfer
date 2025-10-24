//
// Created by pku00991 on 2025/4/27.
//

#ifndef DEEP_COMMUNITY_LOGGER_H
#define DEEP_COMMUNITY_LOGGER_H
#include "ExportInc.h"
#include <string>
#include <cstdarg>
#include "libGlogInc.h"
#include "ExportInc.h"
namespace tff::log {
class DEEP_TFF_API Logger {
public:
  // 初始化日志系统
  static void init(const std::string& appName);
  static void exit();
  // 提供不同级别的日志记录方法，支持 printf 格式化
  static void info(const char* format, ...);
  static void warning(const char* format, ...);
  static void error(const char* format, ...);
  static void fatal(const char* format, ...);

private:
  // 内部辅助函数：格式化字符串
  static std::string formatString(const char* format, va_list args);
};
}

#endif // DEEP_COMMUNITY_LOGGER_H
