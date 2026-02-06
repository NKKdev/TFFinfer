//
// Created by nkk on 2025/10/24.
//

#include "GraphNode.h"

namespace tff::core::graph {
    std::function<tff::kernel::base::OP_CALLBACK_TYPE> GraphNode::forward() {
        const auto dev = device();
        if (dev.empty()) {
            tff::log::Logger::error("Node '%s': no valid device bound", this->_node_metadata._name.c_str());
            return nullptr;
        }
        auto device_id = this->_devices.begin()->first;
        auto device_ptr = this->_devices.begin()->second;
        if (this->_tensor != nullptr && this->_tensor->get_buffer() == nullptr && this->op_type() != TFF_OP_MAP2CPU &&
            this->op_type() != TFF_OP_VIEW && this->op_type() != TFF_OP_MEM_REF && this->op_type() != TFF_OP_RESHAPE) {
            auto buffer = this->_mem_manager_ptr->allocate_memory(
                this->_tensor->get_bytes(), device_id, this->_tensor->memory_type());
            tff::log::Logger::info("device_id: %d,memory type: %d, op %s tensor buffer start offset: %lld end offset: %lld",
                device_id, this->_tensor->memory_type(),this->_node_metadata._name.c_str(),
                buffer.first,buffer.first + this->_tensor->get_bytes());
            this->_tensor->set_buffer_data(buffer.second, this->_tensor->get_bytes(),
                buffer.first);
            this->_tensor->set_allocator(device_ptr->get_device_buffer_allocator(device_id));
            }
        //
        for (auto &consumer : this->output_nodes()) {
            auto event = consumer->event();
            this->_mem_manager_ptr->aquire_memory(device_id,
                this->_tensor->get_external_memory_index(), this->_tensor->get_bytes(),
                event);
        }
        //
        std::vector<std::shared_ptr<core::device::DeviceEvent>> events_list;
        for (auto &input : this->input_nodes()) {
            events_list.push_back(input->event());
            if (input->op_type() != TFF_OP_VIEW) {
                this->add_inputs(input->get_tensor());
            }
        }
        //
        auto params_ptr = this->get_params();
        params_ptr->set_param(this->_tensor);
        params_ptr->set_param(this->_inputs);
        params_ptr->set_param(this->_node_metadata._name);
        params_ptr->set_param(this->_mem_manager_ptr);
        params_ptr->set_param(this->event());
        params_ptr->set_param(events_list);
        params_ptr->set_param(this->stream());

        auto callback = kernel::base::get_op_func(this->_devices.begin()->second,
            this->_op_type, this->data_type());
        return callback;
    };
}