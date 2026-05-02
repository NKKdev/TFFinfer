//
// Created by nkk on 2026/2/11.
//
#include <filesystem>
#include "TFFOPCreatorBase.h"

#include "Builder.h"
#include "device/DeviceManager.h"

namespace tff::kernel::base {
#ifdef _DEBUG
         //
    void save_tensor(const char *filename, std::shared_ptr<core::memory::Tensor> &tensor) {
        FILE *fp = fopen(filename, "wb");
        if (!fp) {
            tff::log::Logger::error("%s: failed to open %s for writing\n", __func__, filename);
            return;
        }

        const uint32_t magic = 0x67676d6c; // 'ggml' in hex
        fwrite(&magic, sizeof(magic), 1, fp);

        const int ne0 = tensor->get_shape()[0];
        const int ne1 = tensor->get_shape()[1];
        const int ne2 = tensor->get_shape()[2];
        const int ne3 = tensor->get_shape()[3];
        int32_t ne[4] = {ne0, ne1, ne2, ne3};
        fwrite(ne, sizeof(int32_t), 4, fp);

        const int nb0 = tensor->get_strides()[0];
        const int nb1 = tensor->get_strides()[1];
        const int nb2 = tensor->get_strides()[2];
        const int nb3 = tensor->get_strides()[3];
        int32_t nb[4] = {nb0, nb1, nb2, nb3};
        fwrite(nb, sizeof(int32_t), 4, fp);

        int32_t type_i32 = (int32_t) tensor->get_data_type();
        fwrite(&type_i32, sizeof(int32_t), 1, fp);

        size_t total_bytes = tensor->get_bytes();

        fwrite(tensor->get_buffer()->ptr(), 1, total_bytes, fp);

        fclose(fp);
        //tff::log::Logger::info("%s: saved tensor '%s' to %s (%zu bytes)\n", __func__, filename, filename, total_bytes);
    }

    void load_tensor(const char *filename, std::shared_ptr<core::memory::Tensor> &tensor) {
        if (!filename) {
             tff::log::Logger::error("%s: filename is null\n", __func__);
            return;
        }
        if (!std::filesystem::exists(std::filesystem::path(filename))) {
            tff::log::Logger::error("%s: filename %s is not exist\n", __func__, filename);
            return;
        }

        FILE *fp = fopen(filename, "rb");
        if (!fp) {
            tff::log::Logger::error("%s: failed to open %s\n", __func__, filename);
            return;
        }

        // 1. Check magic
        uint32_t magic;
        if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != 0x67676d6c) {
            tff::log::Logger::error("%s: invalid magic in %s\n", __func__, filename);
            fclose(fp);
            return;
        }

        // 2. Read ne and nb
        int32_t ne32[4], nb32[4];
        if (fread(ne32, sizeof(int32_t), 4, fp) != 4 || fread(nb32, sizeof(int32_t), 4, fp) != 4) {
           tff::log::Logger::error("%s: failed to read ne/nb\n", __func__);
            fclose(fp);
            return;
        }

        // Convert to int64_t / size_t
        int64_t ne[4];
        size_t nb[4];
        for (int i = 0; i < 4; ++i) {
            ne[i] = (int64_t) ne32[i];
            nb[i] = (size_t) nb32[i];
        }

        // 3. Read type
        int32_t type_i32;
        if (fread(&type_i32, sizeof(int32_t), 1, fp) != 1) {
            tff::log::Logger::error( "%s: failed to read type\n", __func__);
            fclose(fp);
            return;
        }
        tff::core::memory::DataType type = (tff::core::memory::DataType) type_i32;
        if (type < 0 || type >= core::memory::DataType::TFF_DATA_TYPE_COUNT) {
            tff::log::Logger::error("%s: invalid ggml_type %d\n", __func__, type_i32);
            fclose(fp);
            return;
        }
        tensor = std::make_shared<core::memory::Tensor>(type, core::memory::MemoryType::TFF_MEM_TYPE_RESIDENT,
            std::array<int64_t, MAX_TENSOR_DIM>{ne[0], ne[1], ne[2], ne[3]}, false);
        auto device_manager = std::dynamic_pointer_cast<core::device::DeviceManager>(
                   tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                       DEVICE_MANAGER_FLAG,
                       tff::factory::ModuleKeyType(DEVICE_MANAGER_FLAG)));
        auto device = device_manager->get_device(-1);
        tensor->set_allocator(device->get_device_buffer_allocator(-1));
        tensor->allocate();
        // 4. Compute data size and allocate
        size_t data_size = tensor->get_bytes();
        if (!tensor->get_buffer()) {
            tff::log::Logger::error("%s: failed to allocate %zu bytes for data\n", __func__, data_size);
            fclose(fp);
            return;
        }

        if (fread(tensor->get_buffer()->ptr(), 1, data_size, fp) != data_size) {
            tff::log::Logger::error("%s: failed to read data\n", __func__);
            fclose(fp);
            return;
        }
        fclose(fp);

        // printf("%s: loaded raw tensor from %s (shape=[%ld,%ld,%ld,%ld], type=%d, data_size=%zu)\n", __func__, filename,
        //        ne[0], ne[1], ne[2], ne[3], (int) type, data_size);

        return;
    }
    //
    void varify(const char *filename, std::shared_ptr<core::memory::Tensor> &tensor) {
        if (tensor->get_data_type() == core::memory::DataType::TFF_DATA_TYPE_I64) {
            return;
        }
        std::shared_ptr<core::memory::Tensor> tensor_cpu;
        load_tensor(filename,  tensor_cpu);

        std::shared_ptr<core::memory::Tensor> tensor_gpu ;
        if (tensor->get_allocator()->device_type() == core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU) {
            tensor_gpu = std::make_shared<core::memory::Tensor>(tensor);

            void *ptr = (void*)malloc(tensor->get_bytes());
            tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), ptr, tensor->get_bytes(),
                core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2HOST);
            tensor_gpu->set_buffer_data(ptr, tensor->get_bytes());

            if (tensor_cpu != nullptr) {
                auto dim = tensor->dims() > 3 ? 3 : tensor->dims();
                auto num = tensor->get_strides()[dim] / tensor->get_strides()[0];
                bool bRet = true;
                for (int i = 0;i < num; i++) {
                    auto old_value = ((static_cast<char*>(tensor_cpu->get_buffer()->ptr()) + i * tensor_cpu->get_strides()[0]));
                    auto new_value = ((static_cast<char*>(tensor_gpu->get_buffer()->ptr()) + i * tensor_gpu->get_strides()[0]));

                    if (std::memcmp(old_value, new_value, tensor->get_strides()[0]) != 0) {
                        tff::log::Logger::info("filename: %s error at element index: %d, tensor : %p buffer : %p",
                            filename, i,tensor.get(), tensor->get_buffer()->ptr());
                        bRet = false;
                        break;
                    }
                }
                if (bRet) {
                    tff::log::Logger::info("filename : %s verify success tensor : %p buffer : %p",
                    filename, tensor.get(), tensor->get_buffer()->ptr());
                }

            }
            save_tensor(filename, tensor_gpu);
            free(ptr);
            ptr = nullptr;
        }else {
            tensor_gpu = tensor;
            if (tensor_cpu != nullptr) {
                auto dim = tensor->dims() > 3 ? 3 : tensor->dims();
                auto num = tensor->get_strides()[dim] / tensor->get_strides()[0];
                bool bRet = true;
                for (int i = 0;i < num; i++) {
                    auto old_value = ((static_cast<char*>(tensor_cpu->get_buffer()->ptr()) + i * tensor_cpu->get_strides()[0]));
                    auto new_value = ((static_cast<char*>(tensor_gpu->get_buffer()->ptr()) + i * tensor_gpu->get_strides()[0]));

                    if (std::memcmp(old_value, new_value, tensor->get_strides()[0]) != 0) {
                        tff::log::Logger::info("filename: %s error at element index: %d, tensor : %p buffer : %p",
                            filename, i,tensor.get(), tensor->get_buffer()->ptr());
                        bRet = false;
                        break;
                    }
                }
                if (bRet) {
                    tff::log::Logger::info("filename : %s verify success tensor : %p buffer : %p",
                    filename, tensor.get(), tensor->get_buffer()->ptr());
                }

            }
            save_tensor(filename, tensor_gpu);
        }
    }
#endif
    std::function<OP_CALLBACK_TYPE> get_op_func(std::shared_ptr<core::device::MemBufferAllocatorBaseObject> &allocator,
                                                const tff::core::graph::TffOpType &op_type,
                                                const tff::core::memory::DataType &data_type) {
        if (allocator == nullptr) {
            tff::log::Logger::error("op device is invalid!!");
            return nullptr;
        }
        auto it = core::global::TFF_OP_TYPE_MAP.find(op_type);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return nullptr;
        }
        std::string device_tag_name;
        switch (allocator->device_type()) {
            case core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU:
                device_tag_name = std::string(core::device::GPUTag::name());
                break;
            case core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU:
                device_tag_name = std::string(core::device::CPUTag::name());
                break;
            case core::device::TFF_BACKEND_DEVICE_TYPE_UNKNOWN:
                break;
        }
        std::string op_name = std::string(it->second) + std::string("_") + device_tag_name + "_";
        switch (data_type) {
            case tff::core::memory::DataType::TFF_DATA_TYPE_F64:
                op_name = op_name + tff::core::global::get_type_suffix<double>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I32:
                op_name = op_name + tff::core::global::get_type_suffix<int32_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I64:
                op_name = op_name + tff::core::global::get_type_suffix<int64_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I8:
                op_name = op_name + tff::core::global::get_type_suffix<uint8_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F16:
                op_name = op_name + tff::core::global::get_type_suffix<half>();
                break;
            case tff::core::memory::TFF_DATA_TYPE_Q8_0:
                op_name = op_name + tff::core::global::get_type_suffix<Q8_0>();
                break;
            case core::memory::TFF_DATA_TYPE_Q8_0_ALIGNED:
                op_name += tff::core::global::get_type_suffix<Q8_0_ALIGNED>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F32:
            default:
                op_name = op_name + tff::core::global::get_type_suffix<float>();
                break;
        }


        return tff::factory::FunctionFactory::instance()->get_callback<tff::kernel::base::OP_CALLBACK_TYPE>(
            OP_NODE_FLAG,
            op_name);
    }

    void op_before_hook(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto tensor = get_param_value<std::shared_ptr<core::memory::Tensor> >(
            builder::IParamBuilder::CommonParams::Out, para_ptr);
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<core::memory::Tensor> > >(
            builder::IParamBuilder::CommonParams::InputList, para_ptr);
        const auto &name = get_param_value<std::string>(builder::IParamBuilder::CommonParams::Name, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(builder::IParamBuilder::CommonParams::MemManager, para_ptr);
        auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent> >(
            builder::IParamBuilder::CommonParams::Event, para_ptr);
        auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent> > >(
            builder::IParamBuilder::CommonParams::WaitList, para_ptr);
        auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            builder::IParamBuilder::CommonParams::Stream, para_ptr);

        for (auto &wait_event: event_list) {
            if (wait_event == nullptr) {
                continue;
            }
            //tff::log::Logger::info("stream %s wait event %s",stream->name().c_str(), wait_event->name().c_str());
            stream->wait_event(wait_event->get_native_event());
        }
        // tff::log::Logger::info("node %s output_tensor pointer %p",name.c_str(),
        //     static_cast<void *>(tensor.get()));
    }
    void op_after_hook(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto tensor = get_param_value<std::shared_ptr<core::memory::Tensor> >(
           builder::IParamBuilder::CommonParams::Out, para_ptr);
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<core::memory::Tensor> > >(
            builder::IParamBuilder::CommonParams::InputList, para_ptr);
        const auto &name = get_param_value<std::string>(builder::IParamBuilder::CommonParams::Name, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(builder::IParamBuilder::CommonParams::MemManager, para_ptr);
        auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent> >(
            builder::IParamBuilder::CommonParams::Event, para_ptr);
        auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent> > >(
            builder::IParamBuilder::CommonParams::WaitList, para_ptr);
        auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            builder::IParamBuilder::CommonParams::Stream, para_ptr);
        if (event != nullptr) {
            event->record(stream);
        }

        if (mem_buffer_manager_ptr != nullptr) {
            for (auto &input: input_tensors) {
                if (input->get_allocator() == nullptr) {
                    continue;
                }
                auto &device_id = input->get_allocator()->_device_id;
                if (input->get_external_memory_index() != -1) {
                    mem_buffer_manager_ptr->release_memory(device_id, input->get_external_memory_index());
                }
            }
        }
#ifdef _DEBUG
        //tff::log::Logger::info("op %s node computed", name.c_str());
#endif
    }
}
