#pragma once
#include <cstdio>
#include <map>
#include <cstring>
#include <functional>
#include <stdexcept>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include "driver_functions.h"
#include <cuda_fp16.h>
#include "driver_types.h"
#include <pthread.h>
#include <csignal>
#include "Logger.h"
#define TIMEOUT 10000



#define CudaSafeCall(error) tff::core::device::cuda::cuda_safe_call(error, __FILE__, __LINE__, #error)

#define CudaSafeCallEx(call, T) tff::core::device::cuda::cuda_safe_callx<T>([&] { return (call); }, __FILE__, __LINE__, #call)

#define CudaIsSafeCall(call, T) Add(CudaSafeCallEx(call, T))

#define CudaKernelCheck() CudaSafeCall(cudaGetLastError())

namespace tff::core::device::cuda
{
	inline void cuda_safe_call(cudaError error, const char *file, const int line, const char *_code_str)
	{

		if (error != cudaSuccess)
		{
			fprintf(stderr, "cuda error %s : %d %s, api: %s \n", file, line, cudaGetErrorString(error), _code_str);
			tff::log::Logger::error("cuda error %s : %d %s, api: %s \n", file, line, cudaGetErrorString(error), _code_str);
			std::string error_code(_code_str);
			std::string sub_str = error_code.substr(0, error_code.find_first_of("("));
			std::string result_str = sub_str + " " + cudaGetErrorString(error);
			throw result_str.c_str();
		}
	}
	template <typename T, class Fn, class Ret>
	struct FuncPara
	{
		std::function<Ret()> call;
		pthread_cond_t *g_cond;
		pthread_mutex_t *g_mutex;
		std::string code_str;
		Ret error;
	};

	template <typename T, class Fn, class Ret>
	inline void *CallFunc(void *_arg)
	{
		FuncPara<T, Fn, Ret> *para = (FuncPara<T, Fn, Ret> *)(_arg);
		para->error = (Ret)para->call();

		// 发结束信号;
		pthread_mutex_lock(para->g_mutex);
		pthread_cond_signal(para->g_cond);
		pthread_mutex_unlock(para->g_mutex);
	}

	static long long tm_to_ns(struct timespec tm)
	{
		return tm.tv_sec * 1000000000 + tm.tv_nsec;
	}

	static struct timespec ns_to_tm(long long ns)
	{
		struct timespec tm;
		tm.tv_sec = ns / 1000000000;
		tm.tv_nsec = ns - (tm.tv_sec * 1000000000);
		return tm;
	}

	template<class Ret, class...Args>
	auto FnRetType(std::function<Ret(Args...)>)
	{
		return Ret{};
	}

	template<class ErrType>
	auto cudaGetErrorString(ErrType &_error)
	{
		const char *error = nullptr;
		cuGetErrorName(_error, &error);
		return error;
	}
	
	template <typename T, class Fn>
	inline std::string cuda_safe_callx(Fn &&call, const char *file, const int line, const char *_code_str)
	{
		pthread_cond_t g_cond;
		pthread_mutex_t g_mutex;

		pthread_cond_init(&g_cond, NULL);
		pthread_mutex_init(&g_mutex, NULL);

		std::function fn(call);

		using ret_type = decltype(FnRetType(fn));

		tff::core::device::cuda::FuncPara<T, Fn, ret_type> para{fn, &g_cond, &g_mutex, std::string(_code_str)};

		pthread_t threadID = 0;
		int err = pthread_create(&threadID, nullptr, CallFunc<T, Fn, ret_type>, (void *)&para);
		pthread_detach(threadID);

		struct timespec start_tm;
		struct timespec end_tm;
		int timeout_ms = TIMEOUT;

		clock_gettime(CLOCK_REALTIME, &start_tm);
		end_tm = ns_to_tm(tm_to_ns(start_tm) + timeout_ms * 1000000);

		pthread_mutex_lock(&g_mutex);
		if (pthread_cond_timedwait(&g_cond, &g_mutex, &end_tm) == ETIMEDOUT)
		{
			pthread_mutex_unlock(&g_mutex);
			pthread_cancel(threadID);
			void *return_str = nullptr;
			pthread_join(threadID, &return_str);
			pthread_cond_destroy(&g_cond);
			pthread_mutex_destroy(&g_mutex);
			std::string error_code(_code_str);
			std::string sub_str = error_code.substr(0, error_code.find_first_of("("));
			const char *api_name = strdup(sub_str.c_str());
			T t(api_name, "connect time out!");
			throw t;
		}
		else
		{
			pthread_mutex_unlock(&g_mutex);
			pthread_cond_destroy(&g_cond);
			pthread_mutex_destroy(&g_mutex);
			if (para.error != cudaSuccess)
			{
				std::string error_code(_code_str);
				std::string sub_str = error_code.substr(0, error_code.find_first_of("("));
				std::string result_str = sub_str + " " + cudaGetErrorString(para.error);
				const char *api_name = strdup(sub_str.c_str());
				const char *error = strdup(result_str.c_str());
				T t(api_name, error);
				throw t;
			}else if(cudaError_enum::CUDA_SUCCESS != para.error){
				std::string error_code(_code_str);
				std::string sub_str = error_code.substr(0, error_code.find_first_of("("));
				std::string result_str = sub_str + " " + cudaGetErrorString(para.error);
				const char *api_name = strdup(sub_str.c_str());
				const char *error = strdup(result_str.c_str());
				T t(api_name, error);
				throw t;
			}
			else{
				std::string error_code(_code_str);
				std::string sub_str = error_code.substr(0, error_code.find_first_of("("));
				return sub_str;
			}
		}

		
	}
}