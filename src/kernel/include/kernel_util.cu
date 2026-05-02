//
// Created by nkk on 2026/2/11.
//
#include "kernel_util.h"

#include "device/DeviceManager.h"

namespace tff::kernel {
#ifdef _DEBUG
    void varify(std::string &op_name, std::string &filename, std::shared_ptr<core::memory::Tensor> &tensor) {
        switch (tensor->get_data_type()) {
            case core::memory::DataType::TFF_DATA_TYPE_F16: {
                std::vector<float> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                // weight_cpu_result.resize(
                //    256*64);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<half> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);

                float error_raio = 0.0f;
                for (int b = 0; b < tensor->get_shape()[3]; b++) {
                    for (int s = 0; s < tensor->get_shape()[2]; s++) {
                        half *gpu_ptr = weight_gpu_result.data() + b * tensor->get_shape()[2] * tensor->get_shape()[1] *
                                        tensor->get_shape()[0]
                                        + s * tensor->get_shape()[1] * tensor->get_shape()[0];
                        float *cpu_ptr = weight_cpu_result.data() + b * tensor->get_shape()[2] * tensor->get_shape()[1]
                                         * tensor->get_shape()[0]
                                         + s * tensor->get_shape()[1] * tensor->get_shape()[0];
                        for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                            for (int nn = 0; nn < tensor->get_shape()[0]; nn++) {
                                float delta = __half2float(gpu_ptr[mm * tensor->get_shape()[0] + nn]) - __half2float(
                                                  cpu_ptr[
                                                      mm * tensor->get_shape()[0] + nn]);
                                if (fabs(delta) > 0.01f) {
                                    tff::log::Logger::error(
                                        "filename: %s, error:s: %d, m: %d n: %d, delta: %lf, gpu: %lf, cpu: %lf",
                                        filename.c_str(), s, mm, nn, delta,
                                        __half2float(gpu_ptr[mm * tensor->get_shape()[0] + nn]),
                                        cpu_ptr[mm * tensor->get_shape()[0] + nn]);

                                    error_raio++;
                                }
                            }
                        }
                    }
                }

                error_raio /= tensor->get_shape()[3] * tensor->get_shape()[2] * tensor->get_shape()[1] * tensor->
                        get_shape()[0];
                if (error_raio > 0.01) {
                    tff::log::Logger::error("op_name: %s, filename: %s, error_raio: %lf", op_name.c_str(),
                                            filename.c_str(), error_raio);
                    return;
                }
                break;
            }
            case core::memory::DataType::TFF_DATA_TYPE_F32: {
                std::vector<float> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<float> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size(), 0);
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);
                float error_raio = 0.0f;
                for (int b = 0; b < tensor->get_shape()[3]; b++) {
                    for (int s = 0; s < tensor->get_shape()[2]; s++) {
                        float *gpu_ptr = weight_gpu_result.data() + b * tensor->get_shape()[2] * tensor->get_shape()[1]
                                         * tensor->get_shape()[0]
                                         + s * tensor->get_shape()[1] * tensor->get_shape()[0];
                        float *cpu_ptr = weight_cpu_result.data() + b * tensor->get_shape()[2] * tensor->get_shape()[1]
                                         * tensor->get_shape()[0]
                                         + s * tensor->get_shape()[1] * tensor->get_shape()[0];
                        for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                            for (int nn = 0; nn < tensor->get_shape()[0]; nn++) {
                                float delta = gpu_ptr[mm * tensor->get_shape()[0] + nn] - cpu_ptr[
                                                  mm * tensor->get_shape()[0] + nn];
                                if (fabs(delta) > 0.01f || isnan(delta)) {
                                    tff::log::Logger::error(
                                        "filename: %s, error: s: %d, m: %d n: %d, delta: %lf, gpu: %lf, cpu: %lf",
                                        filename.c_str(), s, mm, nn, delta, gpu_ptr[mm * tensor->get_shape()[0] + nn],
                                        cpu_ptr[mm * tensor->get_shape()[0] + nn]);

                                    error_raio++;
                                }
                            }
                        }
                    }
                }

                error_raio /= tensor->get_shape()[3] * tensor->get_shape()[2] * tensor->get_shape()[1] * tensor->
                        get_shape()[0];
                if (error_raio > 0.01) {
                    tff::log::Logger::error("op_name: %s, buffer ptr: %p, filename: %s, error_raio: %lf",
                                            op_name.c_str(), tensor->get_buffer()->ptr(), filename.c_str(), error_raio);
                    return;
                }
                break;
            }
            case core::memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED: {
                std::vector<Q8_0> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] / Q8_0::BLOCK_SIZE * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<Q8_0_ALIGNED> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);
                float error_raio = 0.0f;
                for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                    for (int nn = 0; nn < tensor->get_shape()[0] / Q8_0::BLOCK_SIZE; nn++) {
                        float delta = weight_gpu_result[mm * tensor->get_shape()[0] / Q8_0::BLOCK_SIZE + nn].d -
                                      __half2float(weight_cpu_result[
                                          mm * tensor->get_shape()[0] / Q8_0::BLOCK_SIZE + nn].d);
                        if (fabs(delta) > 0.001f) {
                            // tff::log::Logger::error("filename: %s, error: m: %d n: %d, gpu: %lf, cpu: %lf, delta: %lf", filename.c_str(),
                            //                         mm, nn,
                            //                         weight_gpu_result[mm * tensor->get_shape()[0] / Q8_0::BLOCK_SIZE + nn].d,
                            //                         __half2float(weight_cpu_result[
                            //               mm * tensor->get_shape()[0] / Q8_0::BLOCK_SIZE + nn].d),
                            //                         delta);
                            //throw std::runtime_error("error");
                            error_raio++;
                        }
                    }
                }
                error_raio /= tensor->get_shape()[3] * tensor->get_shape()[2] * tensor->get_shape()[1] * tensor->
                        get_shape()[0];
                if (error_raio > 0.01) {
                    tff::log::Logger::error("filename: %s, error_raio: %lf", filename.c_str(), error_raio);
                    return;
                }
                break;
            }
            default:
                break;
        }

        tff::log::Logger::info("layer node op (%s) varify (%s) buffer ptr: %p, success!",
                               op_name.c_str(), filename.c_str(), tensor->get_buffer()->ptr());
    }
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
}
