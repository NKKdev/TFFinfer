# 十年 C++ 后端 GAP 六个月，写了一个近 3 万行的LLM-TFFInfer推理框架项目解析

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 技术架构](#2-技术架构)
- [3. 核心模块详解](#3-核心模块详解)
- [4. 设计模式与架构思想](#4-设计模式与架构思想)
- [5. 数据流与执行流程](#5-数据流与执行流程)
- [6. 关键技术实现](#6-关键技术实现)
- [7. 性能优化策略](#7-性能优化策略)
- [8. 扩展性与可维护性](#8-扩展性与可维护性)

---

## 1. 项目概述

### 1.1 项目简介

**TFFInfer** 是一个从零自主研发的大语言模型(LLM)推理框架,采用现代 **C++20** 标准开发,支持 **CUDA GPU** 加速。项目实现了完整的 LLM 推理流水线,包括:

- 模型加载(支持 GGUF 格式)
- 计算图构建与优化
- 显存管理与优化
- KV Cache 管理
- 任务调度与并行执行
- 高性能 CUDA Kernel 实现

### 1.2 项目规模

- 代码量:近 30,000 行
- 语言:C++20 + CUDA
- 支持的模型:Qwen3-8B (Q8_0 量化)、GGUF 格式模型
- 编译要求:CMake 3.18+、CUDA 12.x

### 1.3 核心技术栈


| 类别      | 技术选型       | 版本    | 用途                 |
| --------- | -------------- | ------- | -------------------- |
| 编程语言  | C++20          | -       | 核心逻辑、模板元编程 |
| GPU 后端  | CUDA           | 12.x    | GPU 算子实现         |
| 构建系统  | CMake          | 3.18+   | 跨平台构建           |
| 日志系统  | Google glog    | v0.7.1  | 高性能日志记录       |
| 任务调度  | Taskflow       | v3.10.0 | DAG 任务流引擎       |
| 数学优化  | libdivide      | v5.2.0  | 快速除法运算         |
| JSON 解析 | nlohmann/json  | v4.0.0  | 配置文件解析         |
| CUTLASS   | NVIDIA CUTLASS | -       | GEMM 优化库          |

---

## 2. 技术架构

### 2.1 整体架构层次

```
应用层 (Application Layer)
    ↓
运行时层 (Runtime Layer) - LLMInferRuntime、TaskFlowManager、MemManager、KVCache
    ↓
核心层 (Core Layer) - Graph Engine、Tensor/Memory、Device Manager
    ↓
模型层 (Model Layer) - ModelLoader、ModelCreator、ModelDetector
    ↓
算子层 (Kernel Layer) - CUDA Kernels、CPU Kernels、OP Creators
    ↓
基础设施层 (Infrastructure Layer) - ModuleFactory、FunctionFactory、Logger
```

### 2.2 目录结构说明

```
tffinfer/
├── include/                    # 公共头文件(对外 API)
│   ├── ExportInc.h            # 导出宏定义
│   ├── ModuleFactory.h        # 模块工厂(核心注册机制)
│   ├── FunctionFactory.h      # 函数工厂(回调注册)
│   ├── ModelFactory.h         # 模型工厂
│   └── ModuleObject.h         # 模块对象基类
│
├── src/
│   ├── core/                  # 核心引擎层
│   │   ├── device/            # 设备管理(CPU/GPU)
│   │   ├── graph/             # 计算图引擎
│   │   ├── mem/               # 内存管理(Tensor/Memory)
│   │   ├── model/             # 模型层(Loader/Creator/Detector)
│   │   ├── runtime/           # 运行时层(Runtime/Manager)
│   │   ├── quant/             # 量化相关
│   │   └── sampler/           # 采样器
│   │
│   ├── kernel/                # 算子实现层
│   │   ├── cuda/              # CUDA Kernel(FlashAttention/GEMM/RoPE等)
│   │   ├── cpu/               # CPU Kernel(备用)
│   │   └── include/           # Kernel 接口定义
│   │
│   ├── taskgraph/             # 任务调度层
│   ├── factory/               # 工厂实现
│   ├── log/                   # 日志封装
│   └── utils/                 # 工具函数
│
├── test/                      # 测试程序
├── model/                     # 模型文件
└── docs/                      # 文档
```

---

## 3. 核心模块详解

### 3.1 基础设施层 (Infrastructure Layer)

#### 3.1.1 ModuleFactory - 模块工厂模式

**文件位置**: `include/ModuleFactory.h`

**核心功能**:
这是一个高度灵活的模块注册与创建工厂,支持两种创建策略:

- **PROTOTYPE**: 每次创建新实例
- **SINGLETON**: 单例模式,重复使用同一实例

**关键设计**:

```cpp
// 支持多种键类型(字符串或枚举)
using ModuleKeyType = std::variant<std::string, int>;

// 创建者信息结构
template<typename Base>
struct CreatorInfo {
    std::function<std::shared_ptr<Base>()> creator;  // 创建函数
    CreationPolicy policy;                            // 创建策略
};

// 单例实例映射(弱引用,避免内存泄漏)
template<typename Base>
struct SingletonInstanceMap {
    static std::unordered_map<ModuleKeyType, std::weak_ptr<Base>, 
                              ModuleKeyHash, ModuleKeyEqual> instances;
};
```

**注册宏**:

```cpp
// 原型模式注册(每次创建新对象)
REGISTER_MODULE_OBJECT_PROTOTYPE(T, Base, type, key, ...)

// 单例模式注册(全局唯一实例)
REGISTER_MODULE_OBJECT(T, Base, type, key, ...)
```

**使用示例**:

```cpp
// 注册模块
REGISTER_MODULE_OBJECT(DeviceCUDA, DeviceBaseObject, 
                      DEVICE_BACKEND_FLAG, DEVICE_BACKEND_TYPE_CUDA);

// 创建模块
auto device = ModuleFactory::instance()->create_shared<DeviceBaseObject>(
    DEVICE_BACKEND_FLAG, DEVICE_BACKEND_TYPE_CUDA);
```

**优势**:

- ✅ 解耦模块创建与使用
- ✅ 支持运行时动态注册
- ✅ 自动管理生命周期
- ✅ 线程安全(通过锁保护)

---

#### 3.1.2 FunctionFactory - 函数回调工厂

**文件位置**: `include/FunctionFactory.h`

**核心功能**:
提供基于标签的函数注册与调用机制,用于解耦算子实现与调用。

**三层结构**:

```
类型 (flag) → 分组 (group) → 键 (key) → 函数 (function)
```

**关键方法**:

```cpp
// 注册回调函数
register_callback<Func>(flag, key, func)

// 获取回调函数
get_callback<Func>(flag, key)

// 调用回调函数
invoke(flag, key, args...)

// 检查是否存在
has_callback(flag, key)
```

**应用场景**:

- 算子实现的动态绑定
- 不同后端的函数切换(CPU/GPU)
- 插件式架构支持

---

### 3.2 设备管理层 (Device Layer)

#### 3.2.1 DeviceManager - 设备管理器

**文件位置**: `src/core/device/DeviceManager.h`

**核心职责**:

- 统一管理所有计算设备(CPU/GPU)
- 设备初始化与查询
- 设备 ID 映射

**关键数据结构**:

```cpp
class DeviceManager : public ModuleObject {
    // 设备列表
    std::set<std::shared_ptr<DeviceBaseObject>> _devices;
  
    // 设备 ID → 设备对象映射
    std::unordered_map<int, std::shared_ptr<DeviceBaseObject>> _device_map;
};
```

**初始化流程**:

```cpp
DeviceManager() {
    // 1. 创建并初始化 GPU 设备
    auto gpu_device = ModuleFactory::instance()
        ->create_shared<DeviceBaseObject>(DEVICE_BACKEND_FLAG, 
                                          DEVICE_BACKEND_TYPE_CUDA);
    gpu_device->device_init();
    _devices.insert(gpu_device);
  
    // 2. 创建并初始化 CPU 设备
    auto cpu_device = ModuleFactory::instance()
        ->create_shared<DeviceBaseObject>(DEVICE_BACKEND_FLAG, 
                                          DEVICE_BACKEND_TYPE_CPU);
    cpu_device->device_init();
    _devices.insert(cpu_device);
}
```

---

#### 3.2.2 MemBufferAllocator - 内存分配器

**文件位置**:

- `src/core/device/cuda/MemBufferAllocatorCUDA.h`
- `src/core/device/cpu/MemBufferAllocatorCPU.h`

**设计理念**:

- 策略模式:不同设备有不同分配策略
- 统一接口:上层无需关心具体实现
- 资源隔离:每个设备独立管理自己的内存

---

### 3.3 内存与张量层 (Memory & Tensor Layer)

#### 3.3.1 Tensor - 张量类

**文件位置**: `src/core/mem/Tensor.h`

**核心功能**:
Tensor 是框架中最基本的数据单元,封装了:

- 多维数组表示
- 数据类型管理
- 内存生命周期
- Stride 推导

**关键属性**:

```cpp
class Tensor : public std::enable_shared_from_this<Tensor> {
    // 维度信息
    size_t _n_dims;                                    // 有效维度数
    std::array<int64_t, MAX_TENSOR_DIM> _shape;        // 形状
    std::array<int64_t, MAX_TENSOR_DIM> _strides;      // 步长
  
    // 类型信息
    DataType _data_type;                               // 数据类型(FP32/FP16/INT8等)
    size_t _type_size;                                 // 元素大小
    uint32_t _blk_size;                                // 块大小(量化用)
  
    // 内存管理
    MemoryType _memory_type;                           // 内存类型(权重/KV/工作区)
    std::shared_ptr<Memory> _buffer;                   // 底层内存
    bool _use_external;                                // 是否使用外部内存
    int64_t _external_memory_index;                    // 外部内存索引
  
    // 生命周期(用于内存复用优化)
    int _start;                                        // 活跃开始时间点
    int _end;                                          // 活跃结束时间点
    int _priority;                                     // 优先级
  
    // 分配器
    std::shared_ptr<MemBufferAllocatorBaseObject> _allocator;
};
```

**Stride 推导算法**:

```cpp
void stride_infer() {
    _strides[0] = _type_size;
    if (_strides.size() > 1) {
        _strides[1] = _strides[0] * (_shape[0] / _blk_size);
    }
    for (int j = 2; j < _shape.size(); ++j) {
        _strides[j] = _strides[j - 1] * _shape[j - 1];
    }
}
```

**内存分配策略**:

```cpp
// 内部内存(自动管理)
Tensor(DataType type, MemoryType mem_type, shapes, false, allocator);
tensor.allocate();  // 自动分配

// 外部内存(由 MemManager 统一管理)
Tensor(DataType type, MemoryType mem_type, shapes, true, nullptr);
tensor.set_buffer_data(ptr, size, mem_offset);  // 绑定外部内存
```

**访问接口**:

```cpp
// 类型安全的元素访问
template<typename T, typename... Args>
T& at(Args... indices) {
    return *reinterpret_cast<T*>(_buffer->ptr() + compute_offset(indices...));
}

// 计算偏移量
size_t compute_offset(Args... indices) {
    std::array<size_t, num_indices> idxs{static_cast<size_t>(indices)...};
    size_t offset = 0;
    for (size_t i = 0; i < num_indices; ++i) {
        offset += idxs[i] * _strides[i];
    }
    return offset;
}
```

---

### 3.4 计算图引擎 (Graph Engine)

#### 3.4.1 Graph - DAG 计算图

**文件位置**: `src/core/graph/Graph.h`

**核心概念**:
计算图是有向无环图(DAG),节点表示算子,边表示数据依赖。

**关键数据结构**:

```cpp
class Graph : public std::enable_shared_from_this<Graph> {
    // 节点集合
    std::vector<std::shared_ptr<GraphNode>> _output_node;   // 输出节点
    std::vector<std::shared_ptr<GraphNode>> _leafs;         // 叶子节点
    std::vector<std::shared_ptr<GraphNode>> _nodes;         // 所有节点
    std::vector<std::shared_ptr<GraphNode>> _total_nodes;   // 完整节点列表
  
    // 执行顺序
    std::unordered_map<std::shared_ptr<GraphNode>, int> _exec_time;  // 执行时间戳
    std::unordered_map<std::shared_ptr<GraphNode>, size_t> _use_counts;  // 使用次数
  
    // 生命周期分析
    std::unordered_map<std::shared_ptr<GraphNode>, int> _first_use;  // 首次使用时间
    std::unordered_map<std::shared_ptr<GraphNode>, int> _last_use;   // 最后使用时间
  
    // 索引映射
    std::unordered_map<std::shared_ptr<GraphNode>, size_t> _node_to_index;
    std::unordered_map<std::shared_ptr<GraphNode>, size_t> _leaf_to_index;
};
```

**建图流程**:

```cpp
void build_graph(std::shared_ptr<GraphNode> output_node) {
    // 1. 从输出节点反向遍历,收集所有依赖节点
    visit(output_node);
  
    // 2. 拓扑排序,确定执行顺序
    // 3. 识别叶子节点(输入节点)
    // 4. 分析节点生命周期
    analyze_lifetimes();
}

void visit(std::shared_ptr<GraphNode> node) {
    if (_visited.count(node)) return;
    _visited.insert(node);
  
    // 递归访问所有输入节点
    for (auto& input : node->inputs()) {
        visit(input);
    }
  
    // 添加到节点列表(后序遍历保证拓扑序)
    _nodes.push_back(node);
}
```

**生命周期分析**:

```cpp
void analyze_lifetimes() {
    // 为每个节点分配执行时间戳
    for (size_t i = 0; i < _nodes.size(); ++i) {
        _exec_time[_nodes[i]] = i;
    }
  
    // 计算每个张量的首次和最后使用时间
    for (auto& node : _nodes) {
        for (auto& output : node->outputs()) {
            int exec_time = _exec_time[node];
            if (_first_use.find(output) == _first_use.end()) {
                _first_use[output] = exec_time;
            }
            _last_use[output] = exec_time;
        }
    }
}
```

---

#### 3.4.2 GraphNode - 图节点

**文件位置**: `src/core/graph/GraphNode.h`

**节点类型**:

```cpp
enum class GraphNodeType {
    TFF_NODE_WEIGHT,      // 权重节点
    TFF_NODE_INPUT,       // 输入节点
    TFF_NODE_OUTPUT,      // 输出节点
    TFF_NODE_OP,          // 算子节点
    TFF_NODE_HOST,        // Host 节点(CPU)
    TFF_NODE_DEVICE,      // Device 节点(GPU)
    TFF_NODE_MEM_CPY,     // 内存拷贝节点
};
```

---

### 3.5 模型层 (Model Layer)

#### 3.5.1 模型加载架构

**设计模式**: 策略模式 + 工厂模式

**核心类层次**:

```
ModelDetectorBase (检测器基类)
    └── GGUFDetector (GGUF 格式检测)

ModelLoaderBase (加载器基类)
    └── GGUFLoader (GGUF 格式加载)

ModelCreatorBase (创建器基类)
    └── QWen3Creator (Qwen3 模型创建)
    └── LLAMACreator (LLaMA 模型创建)
```

---

#### 3.5.2 ModelDetector - 模型检测器

**文件位置**: `src/core/model/base/ModelDetectorBase.h`

**职责**:
根据模型文件格式自动选择合适的加载器。

```cpp
class ModelDetectorBase {
    // 判断是否匹配当前格式
    virtual bool matches(const std::string& file_format) const = 0;
  
    // 检测器名称
    virtual const char* name() const = 0;
  
    // 创建对应的加载器
    virtual std::shared_ptr<ModelLoaderBase> create_loader() = 0;
};
```

---

#### 3.5.3 ModelLoader - 模型加载器

**文件位置**: `src/core/model/base/ModelLoaderBase.h`

**核心接口**:

```cpp
class ModelLoaderBase : public ModuleObject {
    // 从文件加载模型
    virtual ModelLoadResult load_from_file(
        const std::vector<std::string>& model_files_name,
        const ModelConfig& params) = 0;
  
    // 获取模型上下文
    virtual std::shared_ptr<ModelContext> get_model_ctx() = 0;
  
    // 获取模型架构名称
    virtual const char* get_arch_name() const = 0;
  
    // 获取模型配置
    virtual const ModelConfig& get_model_config() const = 0;
  
    // 获取权重映射
    virtual const std::unordered_map<std::string, ModelWeight>& 
        get_weight_map() const = 0;
  
    // 支持 mmap
    virtual bool supports_mmap() const { return true; }
};
```

---

#### 3.5.4 ModelCreator - 模型创建器

**文件位置**: `src/core/model/base/ModelCreatorBase.h`

**核心职责**:
根据模型架构构建计算图。

**GraphContext - 计算图上下文**:

```cpp
struct GraphContext {
    // 模型超参数
    int _n_layer;           // 层数
    int _n_head;            // 注意力头数
    int _n_head_kv;         // KV 头数(支持 GQA)
    int _n_embd_head;       // 每个头的维度
    int _n_tokens;          // Token 数量
  
    // RoPE 配置
    float _rope_freq_base;
    float _rope_freq_scale;
    RopeType _rope_type;
  
    // 归一化配置
    float _f_norm_rms_eps;
    TFFNormType _norm_type;
  
    // 精度配置
    bool _use_fp16;
    bool _is_fuse;          // 是否融合算子
  
    // 运行时状态
    bool _is_prefill;       // 是否是预填充阶段
    int _seq_id;            // 序列 ID
  
    // 关键张量
    std::shared_ptr<Tensor> _logits;      // 输出 logits
    std::shared_ptr<Tensor> _rope_table;  // RoPE 表
    std::shared_ptr<Tensor> _mask;        // Attention Mask
  
    // 依赖组件
    std::shared_ptr<ModelLoaderBase> _model_loader;
    std::shared_ptr<LLMMemManager> _mem_manager_ptr;
    std::unordered_map<int, std::shared_ptr<LLMKVCache>> _kv_cache_ptr;
};
```

**计算图构建接口**:

```cpp
class ModelCreatorBase {
    // 构建主计算图
    virtual void build_graph(...) = 0;
  
    // 构建 IO 计算图(内存管理)
    virtual void build_mem_graph(...) = 0;
  
    // 设置模型上下文
    virtual void build_model_context(const GraphContext& ctx) = 0;
  
    // 辅助构建方法
    std::shared_ptr<GraphNode> build_norm(...);
    std::shared_ptr<GraphNode> build_mul_mat_node(...);
    std::shared_ptr<GraphNode> build_attn(...);
    std::shared_ptr<GraphNode> build_ffn(...);
    std::shared_ptr<GraphNode> build_kv_cache_store_node(...);
    std::shared_ptr<GraphNode> build_kv_cache_load_node(...);
};
```

---

### 3.6 运行时层 (Runtime Layer)

#### 3.6.1 LLMInferRuntime - 推理运行时(核心入口)

**文件位置**: `src/core/runtime/InferRuntime.h`

**核心职责**:
协调整个推理流程,是所有组件的总控制器。

**主要成员**:

```cpp
class LLMInferRuntime {
    // 模型信息
    ModelConfig _model_config;
    std::shared_ptr<LLMVocabulary> _vocabulary_ptr;
    std::shared_ptr<ModelLoaderBase> _model_loader;
    std::shared_ptr<ModelCreatorBase> _model_creator;
  
    // 计算图
    std::shared_ptr<Graph> _prefill_graph_ptr;       // 预填充图
    std::shared_ptr<Graph> _decode_graph_ptr;        // 解码图
    std::shared_ptr<Graph> _mem_graph_ptr;           // 内存管理图
  
    // 优化器与管理器
    std::shared_ptr<GraphOptimizer> _graph_optimizer;
    std::shared_ptr<LLMMemManager> _mem_manager_ptr;
    std::shared_ptr<LLMTaskFlowManager> _task_manager;
    std::shared_ptr<LLMBatchManager> _llm_batch_manager_ptr;
    std::shared_ptr<DeviceManager> _device_manager;
  
    // KV Cache
    std::unordered_map<int, std::shared_ptr<LLMKVCache>> _kv_cache_ptr;
};
```

**初始化流程**:

```cpp
LLMInferRuntime() {
    // 1. 初始化设备
    init_device();
  
    // 2. 创建内存管理器
    _mem_manager_ptr = ModuleFactory::instance()
        ->create_shared<ModuleObject>(WEIGHT_MEM_BUFFER_MANAGER_FLAG, ...);
  
    // 3. 创建批处理管理器
    _llm_batch_manager_ptr = ModuleFactory::instance()
        ->create_shared<ModuleObject>(BATCH_MANAGER_FLAG, ...);
  
    // 4. 创建任务流管理器
    _task_manager = ModuleFactory::instance()
        ->create_shared<ModuleObject>(TASK_FLOW_MANAGER_FLAG, ...);
  
    // 5. 创建图优化器
    _graph_optimizer = ModuleFactory::instance()
        ->create_shared<ModuleObject>(GRAPH_OPTIMIZER_FALG, ...);
}
```

**推理流程**:

```cpp
bool infer(int n_predict, std::vector<std::string>& generate_str) {
    // 1. Prefill 阶段(处理 prompt)
    prefill(batch);
  
    // 2. Decode 阶段(逐个生成 token)
    for (int i = 0; i < n_predict; ++i) {
        decode(batch, n_predict, generate_str);
    
        // 3. 采样下一个 token
        int32_t token = sample_token();
    
        // 4. 更新 batch
        update_batch(token);
    }
  
    return true;
}
```

---

#### 3.6.2 TaskFlowManager - 任务流管理器

**文件位置**: `src/core/runtime/TaskFlowManager.h`

**核心职责**:
基于 Taskflow 库实现 DAG 任务调度。

```cpp
class LLMTaskFlowManager : public ModuleObject {
    std::shared_ptr<HybridScheduler> _task_scheduler;
  
    // 构建任务调度
    bool build_task_schedule(const TaskType& type, 
                            const std::shared_ptr<Graph>& graph_ptr);
  
    // 运行任务
    void run(const TaskType& type) {
        _task_scheduler->run(type);
    }
};
```

---

#### 3.6.3 MemManager - 内存管理器

**文件位置**: `src/core/runtime/MemManager.h`

**核心职责**:

- 统一管理设备内存池
- 基于生命周期的内存分配与回收
- 内存复用优化

**关键数据结构**:

```cpp
class LLMMemManager : public ModuleObject {
    // 内存块
    struct MemoryBlock {
        int64_t _offset;      // 偏移量
        int64_t _size;        // 大小
        int _free_time;       // 释放时间
        int _ref_count;       // 引用计数
        std::vector<std::shared_ptr<DeviceEvent>> _pending_events;
    };
  
    // 空闲块
    struct FreeBlock {
        int64_t _offset;
        int64_t _size;
    };
  
    // 内存池
    std::unordered_map<int, void*> _mem_buffer_map;              // 设备 → 基址
    std::unordered_map<int, int64_t> _current_offset;            // 设备 → 当前偏移
    std::unordered_map<int, std::set<FreeBlock>> _free_set;      // 设备 → 空闲块集合
    std::unordered_map<int, priority_queue<MemoryBlock>> _memory_heap; // 待释放堆
};
```

**内存分配策略**:

```cpp
std::pair<int64_t, void*> allocate_memory(int64_t size, 
                                          const int& device_id,
                                          const MemoryType& type,
                                          const std::shared_ptr<DeviceEvent>& event) {
    std::lock_guard<std::mutex> lock(_mutex);
  
    // 1. 查找合适的空闲块
    auto& free_set = _free_set[device_id];
    auto it = find_best_fit(free_set, size);
  
    if (it != free_set.end()) {
        // 2. 复用空闲块
        int64_t offset = it->_offset;
        free_set.erase(it);
        void* ptr = static_cast<char*>(_mem_buffer_map[device_id]) + offset;
    
        // 3. 记录引用
        aquire_memory(device_id, offset, size, event);
    
        return {offset, ptr};
    } else {
        // 4. 扩展内存池
        int64_t offset = _current_offset[device_id];
        _current_offset[device_id] += align_up(size, _alignment);
    
        void* ptr = static_cast<char*>(_mem_buffer_map[device_id]) + offset;
        return {offset, ptr};
    }
}
```

**延迟回收机制**:

```cpp
void release_memory(const int& device_id, const int64_t& offset) {
    // 不立即释放,而是加入待释放队列
    // 等待 CUDA Event 完成后真正释放
    _memory_heap[device_id].push(MemoryBlock(offset, size, current_time));
}

void collect(const int& device_id) {
    // 检查已完成的 Event,回收内存
    while (!_memory_heap[device_id].empty()) {
        auto& block = _memory_heap[device_id].top();
        if (all_events_completed(block._pending_events)) {
            _free_set[device_id].insert(FreeBlock{block._offset, block._size});
            _memory_heap[device_id].pop();
        } else {
            break;
        }
    }
}
```

---

#### 3.6.4 KVCache - KV 缓存管理

**文件位置**: `src/core/runtime/KVCache.h`

**核心设计**: Paged Attention

**关键组件**:

**1. KVPage - 缓存页**:

```cpp
struct KVPage {
    std::shared_ptr<Tensor> _k;        // Key 张量
    std::shared_ptr<Tensor> _v;        // Value 张量
    int _n_tokens = 0;                 // 当前 token 数
    bool _is_used = false;             // 是否已使用
};
```

**2. PageManager - 页管理器**:

```cpp
class PageManager {
    std::vector<std::shared_ptr<KVPage>> _pages;  // 所有页
    std::vector<PageID> _free_list;                // 空闲页列表
  
    // 分配页
    PageID allocate() {
        if (_free_list.empty()) return INVALID_PAGE_ID;
        PageID id = _free_list.back();
        _free_list.pop_back();
        _pages[id]->_is_used = true;
        return id;
    }
  
    // 释放页
    void free(PageID id) {
        _pages[id]->_is_used = false;
        _free_list.push_back(id);
    }
  
    // 获取 K/V 张量
    std::shared_ptr<Tensor> get_k(PageID id, ...);
    std::shared_ptr<Tensor> get_v(PageID id, ...);
};
```

**3. LayerKVContext - 层上下文**:

```cpp
class LayerKVContext {
    int _seq_id;                    // 序列 ID
    int _layer_id;                  // 层 ID
    std::vector<PageID> _page_table; // 页表(逻辑页 → 物理页)
    int _num_tokens = 0;            // token 数量
  
    // 添加 token
    bool append_token() {
        if (_num_tokens == get_max_tokens()) {
            PageID new_page = _page_manager->allocate();
            if (new_page == INVALID_PAGE_ID) return false;
            _page_table.push_back(new_page);
        }
        _num_tokens++;
        return true;
    }
  
    // 获取 token 位置
    std::pair<PageID, int> get_location(int token_idx) {
        int page_id = token_idx / PAGE_SIZE;
        int offset = token_idx % PAGE_SIZE;
        return {_page_table[page_id], offset};
    }
};
```

**4. LLMKVCache - KV 缓存主类**:

```cpp
class LLMKVCache {
    struct KVConfig {
        uint32_t _n_layer;          // 层数
        uint32_t _n_head;           // 头数
        uint32_t _n_embd_head;      // 头维度
        int _total_pages;           // 总页数
        int _page_size = 32;        // 每页 token 数
    };
  
    std::shared_ptr<PageManager> _page_manager;
    std::unordered_map<int, std::unique_ptr<LayerKVContext>> _seq_contexts;
  
    // 获取上下文
    LayerKVContext* get_context(int seq_id, int layer_id) {
        int key = make_key(seq_id, layer_id);
        auto it = _seq_contexts.find(key);
        if (it != _seq_contexts.end()) {
            return it->second.get();
        }
    
        auto ctx = std::make_unique<LayerKVContext>(seq_id, layer_id, _page_manager);
        LayerKVContext* ptr = ctx.get();
        _seq_contexts[key] = std::move(ctx);
        return ptr;
    }
  
    // 获取 K/V
    std::shared_ptr<Tensor> get_k(int seq_id, int layer_id, PageID page_id, ...) {
        auto ctx = get_context(seq_id, layer_id);
        return ctx->_page_manager->get_k(page_id, ...);
    }
};
```

**Paged Attention 优势**:

- ✅ 消除内存碎片
- ✅ 支持可变长度序列
- ✅ 高效的内存利用率
- ✅ 支持 Continuous Batching(未来扩展)

---

### 3.7 算子层 (Kernel Layer)

#### 3.7.1 TFFOPCreator - 算子创建器

**文件位置**: `src/kernel/include/TFFOPCreator.h`

**核心功能**:
定义所有支持的算子类型及其参数构建器。

**支持的算子**:

```cpp
// 数据传输
Map2Cpu, MemCpy

// Embedding
Embedding

// 线性变换
Mul, MatMul, QuantMatMul

// 激活函数
Unary (SwiGLU, GeLU, etc.)

// 归一化
Norm, NormW (RMSNorm, LayerNorm)

// 位置编码
Rope

// 注意力
FlashAttn, PagedFlashAttn

// 量化
Quant, DeQuant, QuantAligned, QuantReshape

// 形状操作
Reshape, View

// 二元操作
Add, Binary

// 内存操作
MemRef, Gather, SetRow, GetRow
```

**参数构建器模式**:

```cpp
class FlashAttnBuilder : public OpParamBuilderBase<FlashAttnBuilder> {
    struct Params {
        static constexpr const char* Q = "q_tensor";
        static constexpr const char* K = "k_tensor";
        static constexpr const char* V = "v_tensor";
        static constexpr const char* Mask = "mask";
        static constexpr const char* Out = "out";
    };
  
public:
    template<typename T>
    FlashAttnBuilder& q(T&& value) {
        return set(Params::Q, value);
    }
  
    template<typename T>
    FlashAttnBuilder& k(T&& value) {
        return set(Params::K, value);
    }
  
    // ... 其他参数
};
```

**使用示例**:

```cpp
auto builder = FlashAttnBuilder()
    .q(q_tensor)
    .k(k_tensor)
    .v(v_tensor)
    .mask(mask_tensor)
    .out(output_tensor);

// 创建算子节点
auto node = create_op_node<FlashAttn>(builder);
```

---

#### 3.7.2 Flash Attention 实现

**文件位置**: `src/kernel/cuda/flash_attention.cu`

**核心优化技术**:

**1. Shared Memory Tiling**:

```cuda
// 将 K/V 分块加载到 shared memory
__shared__ half2 k_sm[BLOCK_DIM_LD][BLOCK_DIM_K / 2 + PAD_SIZE];
__shared__ half2 v_sm[BLOCK_DIM_K][BLOCK_DIM_V / 2];
```

**2. Async Copy (Hopper)**:

```cuda
// 异步拷贝全局内存到共享内存
cp_async_cg_16<64>(dst_addr, src_addr);
cp_async_commit();
cp_async_wait_all();
```

**3. Swizzle 避免 Bank Conflict**:

```cuda
template<int B, int M, int S = B>
__device__ int swizzle(const int& offset) {
    const int bit_msk = (1 << B) - 1;
    const int yyy_msk = bit_msk << (M + max(0, S));
    if constexpr (S >= 0) {
        return offset ^ ((offset & yyy_msk) >> S);
    } else {
        return offset ^ ((offset & yyy_msk) << -S);
    }
}
```

**4. RoPE 融合**:

```cuda
// 在加载 K/Q 时直接应用 RoPE
template<const int ELEMENTS_PER_LOAD>
__device__ void rote(const int addr, half2* sm, float2* cos_sin_table) {
    cp_async_wait_all();
    float4 sm_val = val_vec[0];
    half2* rot_value = reinterpret_cast<half2*>(&sm_val);
  
    for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
        half2 val = rot_value[i];
        rot_value[i] = complex_mul_half2(val, __float22half2_rn(cos_sin_table[i]));
    }
}
```

**5. Online Softmax**:

```cuda
// 在线计算 softmax,避免额外 pass
float m_prev = m;
float m_new = max(m, row_max);
float l_prev = l * exp(m_prev - m_new);
float l_new = l_prev + exp(row_max - m_new);

// 更新输出
acc = acc * (l_prev / l_new) + attn * (exp(row_max - m_new) / l_new);
m = m_new;
l = l_new;
```

---

### 3.8 任务调度层 (Task Scheduling Layer)

#### 3.8.1 HybridScheduler - 混合调度器

**文件位置**: `src/taskgraph/include/TaskFlowSchedule.h`

**核心功能**:
基于 Taskflow 实现 CPU-GPU 混合任务调度。

**关键特性**:

```cpp
class HybridScheduler : public ModuleObject {
    // Taskflow 执行器
    std::unordered_map<TaskType, tf::Executor> _executor;
    std::unordered_map<TaskType, tf::Taskflow> _task_flow;
  
    // CUDA Graph 支持
    cudaStream_t _capture_stream;
    cudaGraph_t _graph;
    cudaGraphExec_t _graph_exec;
    bool _use_cuda_graph;
  
    // 多流支持(未来扩展)
    std::vector<cudaStream_t> _gpu_streams;
    std::vector<cudaEvent_t> _sync_events;
};
```

**添加任务**:

```cpp
template<typename F, typename... Args>
tf::Task add_task(const TaskType& type, const std::string& name, 
                  F&& f, Args&&... args) {
    auto bound = [func = std::forward<F>(f),
                  tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        std::apply(func, tup);
    };
    return _task_flow[type].emplace(std::move(bound)).name(name);
}
```

**运行任务**:

```cpp
void run(const TaskType& type) {
    if (_use_cuda_graph && _graph_exec) {
        // 使用 CUDA Graph 执行
        cudaGraphLaunch(_graph_exec, _capture_stream);
    } else {
        // 使用 Taskflow 执行
        _future[type] = _executor[type].run(_task_flow[type]);
    }
}

void wait_until_completion(const TaskType& type) {
    if (!_use_cuda_graph) {
        _future[type].wait();
    } else {
        cudaStreamSynchronize(_capture_stream);
    }
}
```

---

## 4. 设计模式与架构思想

### 4.1 使用的设计模式

#### 4.1.1 工厂模式 (Factory Pattern)

- **ModuleFactory**: 模块对象的创建与管理
- **FunctionFactory**: 函数回调的注册与调用
- **ModelFactory**: 模型加载器的选择与创建

**优势**:

- 解耦对象创建与使用
- 支持运行时动态注册
- 便于扩展新模块

---

#### 4.1.2 策略模式 (Strategy Pattern)

- **DeviceBaseObject**: CPU/GPU 设备策略
- **MemBufferAllocator**: 不同设备的内存分配策略
- **ModelDetector**: 不同格式的模型检测策略

**优势**:

- 算法可互换
- 符合开闭原则
- 易于测试

---

#### 4.1.3 观察者模式 (Observer Pattern)

- **GraphNode 依赖关系**: 节点间的输入输出依赖
- **DeviceEvent**: 异步事件通知

---

#### 4.1.4 单例模式 (Singleton Pattern)

- **ModuleFactory::instance()**
- **FunctionFactory::instance()**
- **DeviceManager**(通过工厂创建单例)

---

#### 4.1.5 建造者模式 (Builder Pattern)

- **OpParamBuilderBase**: 算子参数构建
- 各种 Builder(Map2CpuBuilder, FlashAttnBuilder, etc.)

**优势**:

- 参数设置清晰
- 支持链式调用
- 编译期类型检查

---

### 4.2 架构思想

#### 4.2.1 分层架构 (Layered Architecture)

```
应用层 → 运行时层 → 核心层 → 模型层 → 算子层 → 基础设施层
```

**优势**:

- 职责清晰
- 降低耦合
- 便于维护和测试

---

#### 4.2.2 依赖注入 (Dependency Injection)

通过工厂模式和构造函数注入依赖:

```cpp
LLMInferRuntime() {
    _mem_manager_ptr = ModuleFactory::instance()
        ->create_shared<...>(...);
    _task_manager = ModuleFactory::instance()
        ->create_shared<...>(...);
}
```

---

#### 4.2.3 面向接口编程 (Programming to Interface)

大量使用虚函数和抽象基类:

```cpp
class ModelLoaderBase { /* 纯虚接口 */ };
class DeviceBaseObject { /* 纯虚接口 */ };
class ModelCreatorBase { /* 纯虚接口 */ };
```

---

#### 4.2.4 RAII (Resource Acquisition Is Initialization)

- `std::shared_ptr` 管理资源生命周期
- `std::lock_guard` 管理锁
- Tensor/Memory 析构时自动释放资源

---

## 5. 数据流与执行流程

### 5.1 完整推理流程

```
1. 初始化阶段
   ├─ LLMInferRuntime 构造
   ├─ 设备初始化 (DeviceManager)
   ├─ 管理器创建 (MemManager, TaskFlowManager, BatchManager)
   └─ 图优化器创建
   
2. 模型加载阶段
   ├─ 加载模型文件 (ModelLoader)
   ├─ 解析元数据 (架构、超参数)
   ├─ 构建层映射 (Layer Map)
   ├─ 加载词表 (Vocabulary)
   └─ 加载权重 (支持 mmap)
   
3. 运行时初始化
   ├─ 初始化 KV Cache (LLMKVCache)
   ├─ 构建计算图 (ModelCreator)
   │   ├─ Prefill Graph
   │   ├─ Decode Graph
   │   └─ Memory Graph
   ├─ 图优化 (GraphOptimizer)
   │   ├─ 算子融合
   │   ├─ 死代码消除
   │   └─ 设备分配
   ├─ 内存规划 (MemManager)
   │   ├─ 生命周期分析
   │   └─ 内存池分配
   └─ 任务流构建 (TaskFlowManager)
   
4. Prefill 阶段 (处理 Prompt)
   ├─ Tokenize 输入文本
   ├─ Embedding 查找
   ├─ 前向传播 (多层 Transformer)
   │   ├─ RMS Norm
   │   ├─ Self Attention (RoPE + Flash Attention)
   │   │   ├─ Q/K/V Projection
   │   │   ├─ RoPE 编码
   │   │   ├─ KV Cache Store
   │   │   └─ Flash Attention
   │   ├─ FFN (SwiGLU)
   │   │   ├─ Gate Projection
   │   │   ├─ Up Projection
   │   │   ├─ SwiGLU Activation
   │   │   └─ Down Projection
   │   └─ Residual Connection
   └─ 输出 Logits
   
5. Decode 阶段 (逐 Token 生成)
   ├─ 采样下一个 Token (Sampler)
   ├─ 更新 Batch
   ├─ 前向传播 (单层,仅最后一个 token)
   ├─ KV Cache Update
   └─ 重复直到生成足够 Token 或遇到 EOS
   
6. 清理阶段
   ├─ 释放 KV Cache
   ├─ 重置内存池
   └─ 清理临时张量
```

---

### 5.2 计算图执行流程

```
TaskFlowManager::run(TFF_TASK_TYPE_INFER)
    ↓
HybridScheduler::run()
    ↓
遍历 Graph Nodes (拓扑序)
    ↓
对于每个 Node:
    ├─ 检查依赖是否就绪
    ├─ 分配输入张量内存
    ├─ 执行 Kernel (GPU/CPU)
    │   ├─ Launch CUDA Kernel
    │   └─ Record CUDA Event
    ├─ 标记输出张量就绪
    └─ 计划内存释放 (延迟回收)
    ↓
等待所有任务完成
    ↓
返回结果
```

---

### 5.3 内存管理流程

```
Tensor 请求内存
    ↓
MemManager::allocate_memory()
    ↓
查找空闲块 (Best Fit)
    ├─ 找到 → 复用
    └─ 未找到 → 扩展内存池
    ↓
记录引用 (aquire_memory)
    ↓
返回内存指针
    ↓
... 计算使用 ...
    ↓
Tensor 释放 / 超出生命周期
    ↓
MemManager::release_memory()
    ↓
加入待释放队列 (带 Event)
    ↓
collect() 检查 Event 完成
    ↓
真正释放 → 加入空闲块集合
```

## 6. 总结与展望

### 6.1 项目亮点

✅ **完整的推理框架**: 从模型加载到推理执行的全链路实现
✅ **高性能 CUDA Kernel**: Flash Attention、量化 GEMM 等优化
✅ **先进的内存管理**: 生命周期分析、延迟回收、内存复用
✅ **Paged KV Cache**: 高效管理注意力缓存
✅ **灵活的架构**: 工厂模式、策略模式,易于扩展
✅ **现代 C++**: C++20 特性、智能指针、模板元编程

---

### 6.2 当前限制

⚠️ **仅支持 GGUF 格式**: 需要扩展更多格式
⚠️ **单卡推理**: 尚未实现多卡并行
⚠️ **静态 Batch**: 缺少 Continuous Batching
⚠️ **有限模型支持**: 目前仅 Qwen3-8B

---

### 9.3 未来计划

🚀 **短期目标** (3-6 个月):

- [ ]  支持更多模型 (LLaMA-3, Mistral)
- [ ]  实现 Continuous Batching
- [ ]  添加更多量化格式 (INT4, FP8)
- [ ]  完善文档和示例
- [ ]  另外结合最近的工作内容，未来可能会优先支持LDM结构化数据大模型->与LLM,世界模型等并列的AGI三大领域之一的LDM模型

🚀 **中期目标** (6-12 个月):

- [ ]  多卡并行推理
- [ ]  支持更多硬件后端 (AMD ROCm, Intel GPU)
- [ ]  实现 Speculative Decoding
- [ ]  集成 vLLM 风格的调度器

🚀 **长期愿景** (1-2 年):

- [ ]  成为各位推理爱好者的学习项目，我们一起努力

---

## 附录

### A. 关键文件索引


| 模块            | 关键文件                                   | 行数 |
| --------------- | ------------------------------------------ | ---- |
| 运行时          | `src/core/runtime/InferRuntime.h`          | 269  |
| 计算图          | `src/core/graph/Graph.h`                   | 168  |
| 张量            | `src/core/mem/Tensor.h`                    | 485  |
| KV Cache        | `src/core/runtime/KVCache.h`               | 476  |
| 内存管理        | `src/core/runtime/MemManager.h`            | 277  |
| 任务调度        | `src/taskgraph/include/TaskFlowSchedule.h` | 124  |
| 算子创建        | `src/kernel/include/TFFOPCreator.h`        | 1645 |
| Flash Attention | `src/kernel/cuda/flash_attention.cu`       | 822  |
| 模块工厂        | `include/ModuleFactory.h`                  | 331  |

---

### B. 参考资料

1. **Flash Attention**: [Dao et al., 2022](https://arxiv.org/abs/2205.14135)
2. **Paged Attention**: [Kwon et al., 2023](https://arxiv.org/abs/2309.06180)
3. **GGUF Format**: [ggml-org/gguf](https://github.com/ggml-org/gguf)
4. **Taskflow**: [taskflow/taskflow](https://github.com/taskflow/taskflow)
5. **CUTLASS**: [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

---

### C. 联系方式

- **项目地址**:
  - Gitee: https://gitee.com/NKK_Ovit/tffinfer.git
  - GitHub: https://github.com/NKKdev/TFFinfer.git
- **作者**: NKK
- **License**: MIT

---

**文档版本**: v1.0
**最后更新**: 2026-04-19
**适用版本**: TFFInfer v0.x
