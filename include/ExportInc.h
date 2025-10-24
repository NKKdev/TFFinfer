//
// Created by nkk on 2025/4/25.
//

#ifndef DEEP_TFF_EXPORTINC_H
#define DEEP_TFF_EXPORTINC_H
#if defined(_WIN32) || defined(_WIN64)
#ifdef _DEEP_TFF_EXPORTS
#define DEEP_TFF_API __declspec(dllexport)
#else
#define DEEP_TFF_API __declspec(dllimport)
#endif
#else
#define DEEP_TFF_API __attribute__((visibility("default")))
#endif
#endif // DEEP_TFF_EXPORTINC_H

#include <functional>
#include <string>
namespace tff {
template <typename R, typename... Args>
using UECallback = std::function<R(Args...)>;
}