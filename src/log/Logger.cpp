//
// Created by pku00991 on 2025/4/27.
//

#include "Logger.h"
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
namespace tff::log {
class CustomLogSink : public google::LogSink {
public:
  void send(google::LogSeverity severity, const char *full_filename,
            const char *base_filename, int line, const google::LogMessageTime& time,
            const char *message, size_t message_len) override {
    std::ostringstream time_stream;
    time_stream << std::put_time(&time.tm(), "%Y-%m-%d %H:%M:%S");

    std::cout << "[" << time_stream.str() << "] "
              << google::GetLogSeverityName(severity) << " " << base_filename
              << ":" << line << " " << message << std::endl;
  }
};

// 初始化日志系统
void Logger::init(const std::string &appName) {
  std::string logDir = "./logs";
  if (!std::filesystem::exists(logDir)) {
    std::filesystem::create_directories(logDir);
  }
  // 初始化 glog，包括设置最小输出级别和日志文件路径等
  google::InitGoogleLogging(appName.c_str());

  // 设置日志输出到文件，默认情况下 glog 会将日志输出到标准错误
  google::SetLogDestination(
      google::GLOG_INFO,
      (logDir + "/info_").c_str()); // 设置所有 INFO 及以上级别的日志输出路径
  google::SetLogDestination(
      google::GLOG_WARNING,
      (logDir + "/warning_").c_str()); // 设置所有 INFO 及以上级别的日志输出路径
  google::SetLogDestination(
      google::GLOG_ERROR,
      (logDir + "/error_").c_str()); // 设置所有 INFO 及以上级别的日志输出路径
  FLAGS_stderrthreshold = google::GLOG_FATAL;
  CustomLogSink *custom_log_sink = new CustomLogSink();
  google::AddLogSink(custom_log_sink);
}
//
void Logger::exit() { google::ShutdownGoogleLogging(); }
// 辅助函数：格式化字符串
std::string Logger::formatString(const char *format, va_list args) {
  // 首先确定需要的缓冲区大小
  va_list args_copy;
  va_copy(args_copy, args);
  int size = vsnprintf(nullptr, 0, format, args_copy) + 1; // +1 用于 '\0'
  va_end(args_copy);

  if (size <= 0) {
    return "[FORMAT ERROR]";
  }

  // 分配缓冲区并格式化字符串
  std::unique_ptr<char[]> buffer(new char[size]);
  vsnprintf(buffer.get(), size, format, args);

  return std::string(buffer.get());
}

// 日志记录方法，支持 printf 格式化
void Logger::info(const char *format, ...) {
  va_list args;
  va_start(args, format);
  std::string message = formatString(format, args);
  va_end(args);

  LOG(INFO) << message;
}
void Logger::warning(const char *format, ...) {
  va_list args;
  va_start(args, format);
  std::string message = formatString(format, args);
  va_end(args);

  LOG(WARNING) << message;
}

void Logger::error(const char *format, ...) {
  va_list args;
  va_start(args, format);
  std::string message = formatString(format, args);
  va_end(args);

  LOG(ERROR) << message;
}

void Logger::fatal(const char *format, ...) {
  va_list args;
  va_start(args, format);
  std::string message = formatString(format, args);
  va_end(args);

  LOG(FATAL) << message;
}
}
