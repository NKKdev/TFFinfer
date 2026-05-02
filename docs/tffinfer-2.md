# 十年 C++ 后端 GAP 六个月，写了一个近 3 万行的LLM-TFFInfer推理框架项目解析(三)

## 1. 为什么第一节讲「模型加载」

推理框架的入口不是算子，而是 **如何把磁盘上的权重与元数据变成程序里可用的张量与超参数**。

- **算子与图**依赖：层数、隐藏维度、头数、RoPE 维度、数据类型等，几乎都来自模型文件或随附配置。
- **内存与 I/O 策略**依赖：是否 `mmap`、对齐、多文件分片等，在加载阶段就要定调。
- **后续图构建**（`ModelCreator`、各层 `Creator`）会读取 `ModelLoader` 暴露的 `ModelConfig` 与 `weight_map`。

因此，把加载链路理清，等于拿到了整张推理流水线的「地基」。

---

## 2. GGUF 是什么（与本项目的关系）

**GGUF**（GPT-Generated Unified Format）是社区常用的单文件（或可多分片）LLM 权重容器，特点包括：

- 魔数与版本头，便于快速校验；
- **KV 元数据区**：字符串键 + 类型化值（含数组），存放架构名、`n_layer`、`embedding_length` 等；
- **张量信息区**：每个张量的名字、维度、GGML 类型、在数据区中的偏移；
- **数据区**：大块连续二进制，通常按对齐要求排列。

TFFInfer 在 `GGUFDef.h` 中定义了与 GGUF 规范对齐的魔数、默认版本与类型枚举；解析主逻辑在 `GGUFLoader.cpp`。

---

## 3. 运行时如何「接上」加载器（总览）

推理侧入口是 `LLMInferRuntime::load_model`（`InferRuntime.cu`）：

1. 用**文件扩展名**在注册表里查找格式检测器：`ModelDetectorRegistry::find_dector(get_file_ext(...))`。
2. 检测器创建对应格式的 **`ModelLoaderBase` 实现**（GGUF 即 `GGUFLoader`）。
3. 调用 `load_from_file`，完成**元数据与张量描述**的解析（本节重点）。
4. 随后 `build_model_creator`、`load_hparams`、`load_vocab`、`build_layers` 等把加载结果交给图与设备（后续课程展开）。

**格式注册**采用静态注册宏（与工厂模块键绑定），例如：

- `GGUFDetector`：`REGISTER_MODULE_OBJECT(..., MODEL_DETECTOR_FLAG, "gguf")`，`matches` 判断扩展名是否为 `gguf`。
- `GGUFLoader`：`REGISTER_MODULE_OBJECT(..., MODEL_LOADER_FLAG, "gguf")`，供工厂按键创建。

这种「检测器 + 加载器」分离的好处是：将来增加新格式时，可独立注册新的 `ModelDetectorBase` / `ModelLoaderBase`，而不必改运行时主流程。

---

## 4. 文件访问层：`FileLoader` 与 `FileMMap`

### 4.1 `FileLoader`

`FileLoader`（`FileLoader.h`）封装 `fopen` / `fread` / `fseek` / `ftell`：

- 构造时打开文件并记录 `_file_size`；
- 提供模板 `read(T&)`、`read(std::string&)`（先读 8 字节长度再读正文）、`read(std::vector<T>&, n)` 等；
- `file_aligned(alignment)`：将读指针按对齐边界推进（GGUF 数据区前常用）。

这是 **顺序解析 GGUF 文件** 的基础。

### 4.2 `FileMMap`（可选）

当 `ModelConfig::_use_mmap` 为真时，`GGUFLoader::load` 会为每个模型文件构造 `FileMMap`：

- 使用文件的 `fileno` 做 `mmap(..., PROT_READ, MAP_SHARED, ...)`；
- `posix_fadvise(..., POSIX_FADV_SEQUENTIAL)` 提示顺序读；
- 根据 CPU 设备数量判断是否 **NUMA**：多 CPU 时用 `POSIX_MADV_RANDOM` 等策略，并避免 `MAP_POPULATE` 的预取路径（代码里 `numa` 为真时 `prefetch = 0`）。

**注意**：当前 `GGUFLoader::load` 在打开 `mmap` 后，**解析仍主要通过 `FileLoader` 的 `fread`** 完成；`mmap` 映射更多是为后续「零拷贝或按需从映射区取权重」预留（具体权重搬运见 IO 任务图，第 7 节简述）。理解这一点有助于区分「解析阶段」与「权重上设备阶段」。

---

## 5. `GGUFLoader` 解析流水线（与代码一一对应）

公开入口 `load_from_file` 保存用户传入的 `ModelConfig`，再调用内部 `load`。

对**每一个**模型文件路径，`load` 依次执行：


| 步骤 | 函数                              | 作用                                                                    |
| ---- | --------------------------------- | ----------------------------------------------------------------------- |
| 0    | 构造`FileLoader`，可选 `FileMMap` | 打开文件                                                                |
| 1    | `check_file`                      | 校验前 4 字节为`GGUF` 魔数                                              |
| 2    | `load_header`                     | 读版本、张量个数`_n_tensors`、KV 个数 `_n_kv`                           |
| 3    | `load_kv_meta`                    | 循环读取每个 KV：键名、类型、值（或数组）写入`ModelContext::_kv`        |
| 4    | `load_model_config`               | 从`_kv` 抽取推理需要的字段填入 `ModelConfig`                            |
| 5    | `load_tensor_info`                | 读每个张量的名字、维度、类型、**文件内偏移**，并构建 `ModelWeight` 映射 |

任一步失败会短路返回 `false`，`load_from_file` 映射为 `ModelLoadResult::FAILED`。

### 5.1 魔数校验

`check_file` 读取 4 字节与 `GGUF_MAGIC`（`GGUFDef.h` 中的 `"GGUF"`）逐字节比较，失败则打印可读字符便于排查损坏或错文件。

### 5.2 头与版本

`load_header` 将版本读入 `gguf_ctx->_version`，并调用 `ModelContext::check_version()`：

- 拒绝 **GGUF v1**；
- 拒绝高于 `GGUF_VERSION`（当前为 3）的文件；
- 对异常大的 version 值报错（提示可能大小端错误）。

这保证了后续按固定布局读 KV/张量时不会 silent corruption。

### 5.3 KV 元数据

对每个 KV：

1. `read(key)` 得到字符串键；
2. 检查 `_kv` 中是否已存在同名键（重复则失败）；
3. `read(meta_type)`；若为 `TFF_GGUF_TYPE_ARRAY`，再读元素类型与 `num_elements`；
4. 通过宏展开的 `switch` + `handle_gguf_kv<T>` → `load_array_meta`，把值写入 `_kv`。

`ModelContext::KVValue` 使用 `std::variant` 容纳标量与各类型 `std::vector<...>`，与 GGUF 类型系统对应。

### 5.4 从 KV 到 `ModelConfig`：`load_model_config`

`load_model_config` 使用宏 `LOAD_KEY_VALUE` / `LOAD_KEY_VALUES`（定义在 `BaseDefine.h`），按枚举键（如 `LLM_KV_GENERAL_ARCHITECTURE`、`LLM_KV_EMBEDDING_LENGTH`、`LLM_KV_BLOCK_COUNT` 等）从 `ctx->_kv` 取值，写入 `_model_config` 的成员，例如：

- 架构名 → `_arch_name`；
- `embedding_length` → `_n_embd`；
- `block_count` → `_n_layer`；
- 注意力头相关数组 → `_n_head_arr`、`_n_head_kv_arr`；
- FFN 维度数组 → `_n_ff_arr`；
- RMS norm epsilon → `_f_norm_rms_eps`；
- MoE 相关 expert 数量等。

这些字段直接驱动后面 `build_model_creator` 里对 `ModelCreatorBase::_graph_ctx` 的填充（如 `_n_layer`、`_n_head`、`_n_embd_head` 等）。

### 5.5 张量信息与 `ModelWeight`：`load_tensor_info`

对每个张量索引 `i`：

1. 读名字，保证不与已有 `_tensor_info` 重名；
2. 读 `n_dims`，要求在 `1..4`（否则抛异常），再读各维填入 `shapes`；
3. 读 `data_type` 转为 `memory::DataType`，构造 `Tensor`（`MemoryType::TFF_MEM_TYPE_RESIDENT`，带 shape）；
4. 调用 `get_model_tensor_type(tensor_name)` 设置 **语义张量类型**（如某层 Q/K/V/O 权重），供图匹配；
5. 读 GGUF 规范中的 **`_offset`**（注意：此处是**数据区内相对偏移**，在后续循环中会转成「逻辑累计」）。

随后：

- `file_aligned(gguf_ctx->_alignment)`，把文件指针对齐到数据区起点；
- 用 `_offset == gguf_ctx->_size` 的约定，把每个张量的偏移**重算**为在连续 padding 布局下的累计偏移，并更新 `_size`、`_max_tensor_byte_size`；
- 为每个张量构造 `ModelWeight`：文件索引 `_idx`、绝对文件偏移 `_offs = gguf_ctx->_offset + _offset`、对齐后字节数 `_alignment_size`、真实元素字节 `_byte_size`、共享的 `_tensor_ptr`。

最终 `get_weight_map()` 返回 `std::unordered_map<std::string, ModelWeight>`，供各层创建器按名字取权重描述。

### 5.6 张量名的语义：`get_model_tensor_type`

实现会去掉 `.weight` 后缀，按 `_model_config._arch_name` 在 `LLM_TENSOR_NAMES` 中查找该架构的「名字模式 → `ModelTensorType`」表，用 `matches_pattern` 支持 `%d` 占位（层号）。匹配失败则为 `LLM_TENSOR_TYPE_UNKNOWN`。

这意味着：**GGUF 里字符串名字** 被映射为框架内部的**枚举类型**，图构建时不必硬编码完整字符串。

---

## 6. 核心数据结构小结

- **`ModelContext`**：一次加载过程的「文件级上下文」——KV 表、张量信息列表、对齐、数据区总大小、可选 `_data_memory_ptr`（见下）。
- **`ModelConfig`**：面向推理的运行时超参子集（用户参数 + 从 KV 解析出的字段合并使用）。
- **`ModelWeight`**：连接「逻辑张量」与「文件中的字节位置」：哪个分片文件、偏移多少、多大一块。

---

## 7. 权重字节何时进内存 / 设备（与第一节的边界）

`GGUFLoader` 中还实现了 `load_tensor_data`：在成员 `_alloc` 为真时，通过工厂创建 CPU 内存分配器，一次性分配 `_size` 字节，`fread` 整块数据区，再对每个 `Tensor` `set_buffer_data`。

在当前仓库的默认路径里，`load()` **没有**调用 `load_tensor_data`，且 `_alloc` 默认为 `false`。实际推理中，`LLMInferRuntime::init_runtime_context` 会调用 `load_tensor_data()`，其实现是 **构建 IO 类型的 Taskflow 任务并执行**（`InferRuntime.cu`），把权重搬运与设备侧分配放到调度图里完成。

**第一节只需建立概念**：

- **解析阶段**（本节）：魔数、头、KV、张量元数据、`weight_map` —— 已基本完成；
- **搬运阶段**（后续「内存 / IO 图」课程）：按图任务把数据从文件或 mmap 拷到 CPU/GPU buffer。

---

## 8. 学习本节时建议打开的源码文件


| 文件                                          | 建议关注点                                                     |
| --------------------------------------------- | -------------------------------------------------------------- |
| `src/core/runtime/InferRuntime.cu`            | `load_model`、`build_model_creator`                            |
| `src/core/model/base/ModelDetectorRegistry.h` | `find_dector`                                                  |
| `src/core/model/gguf/GGUFDetector.h`          | `matches`、`create_loader`                                     |
| `src/core/model/base/ModelLoaderBase.h`       | 加载器抽象接口                                                 |
| `src/core/model/gguf/GGUFLoader.cpp` / `.h`   | 完整 GGUF 解析流程                                             |
| `src/core/model/FileLoader.h`                 | `FileLoader`、`FileMMap`                                       |
| `src/core/model/GGUFDef.h`                    | 魔数、版本、类型映射宏                                         |
| `src/core/model/BaseDefine.h`                 | `ModelConfig`、`ModelContext`、`ModelWeight`、`LOAD_KEY_VALUE` |

---

## 9. 自测与思考题

1. 若 GGUF 文件魔数正确但 version 为 1，`check_version` 会如何表现？对上层 `load_from_file` 返回值有何影响？
2. `load_tensor_info` 中为何在读完所有张量元数据后还要 `file_aligned`？若省略可能造成什么问题？
3. 若扩展名不是 `gguf`，`ModelDetectorRegistry::find_dector` 可能返回什么？运行时应在何处做判空与错误提示？（可对照 `load_model` 当前实现思考改进点。）
4. `ModelWeight` 里同时保留 `_byte_size` 与 `_alignment_size` 的原因可能是什么？（提示：SIMD / DMA / kernel 对齐读取。）

---

## 10. 下一节预告

**第 2 节** 建议主题：**`ModelCreator` 与计算图构建** —— 在已有 `ModelConfig` 与 `weight_map` 的前提下，如何按架构（如 Qwen3）实例化各层算子节点，并把权重绑定到图上。

---

**相关：** 分模块、分层可直接渲染的架构图（Mermaid + ASCII）见同目录 [`course-01-architecture-diagrams.md`](course-01-architecture-diagrams.md)；PPT 页序与要点提纲见 [`course-01-ppt-architecture-outline.md`](course-01-ppt-architecture-outline.md)。

---

*文档版本：与仓库 `GGUFLoader`、`InferRuntime` 当前实现对齐；若代码变更请以源码为准。*
