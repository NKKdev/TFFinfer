//
// Created by nkk on 2025/10/24.
//

#include "GraphNode.h"

#include "device/DeviceManager.h"
#include "include/Builder.h"
#include "include/TFFOPCreator.h"

namespace tff::core::graph {
    std::function<tff::kernel::base::OP_CALLBACK_TYPE> GraphNode::forward() {
        if (this->_builder == nullptr) {
            log::Logger::error("Node '%s': no valid builder found", this->_node_metadata._name.c_str());
            return nullptr;
        }
        //
        this->prepare_params();
        if (this->_tensor->get_allocator() == nullptr) {
            log::Logger::error("Node '%s': no valid memory allocator found", this->_node_metadata._name.c_str());
            return nullptr;
        }
        auto device_manager = std::dynamic_pointer_cast<device::DeviceManager>(
            tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                DEVICE_MANAGER_FLAG,
                tff::factory::ModuleKeyType(DEVICE_MANAGER_FLAG)));
        auto device_id = this->_tensor->get_allocator()->_device_id;
        if (device_id == -1) {
            this->bind_devices({{device_id, device_manager->get_device(device_id)}});
        } else {
            this->bind_devices({{device_id, device_manager->get_device(device_id)}});
        }
        if (this->op_type() != TFF_OP_MAP2CPU &&
            this->op_type() != TFF_OP_VIEW && this->op_type() != TFF_OP_MEM_REF && this->op_type() !=
            TFF_OP_RESHAPE) {
            auto buffer = this->_mem_manager_ptr->allocate_memory(
                this->_tensor->get_bytes(), device_id, this->_tensor->memory_type(), this->event());
            this->_tensor->set_buffer_data(buffer.second, this->_tensor->get_bytes(),
                                           buffer.first);
        }

        //
        std::vector<std::shared_ptr<core::device::DeviceEvent> > events_list;
        for (auto &input: this->input_nodes()) {
            events_list.push_back(input->event());
            if (input->op_type() != TFF_OP_VIEW) {
                this->add_inputs(input->get_tensor());
            }
            if (input->get_tensor() == nullptr) {
                continue;
            }
            if (input->op_type() != TFF_OP_MEM_REF) {
                auto event = this->event();
                // tff::log::Logger::info("current node %s aquired by node %s ", input->name().c_str(),
                //     this->name().c_str());
                auto input_device_id = input->get_tensor()->get_allocator()->_device_id;
                this->_mem_manager_ptr->aquire_memory(input_device_id,
                                                      input->get_tensor()->get_external_memory_index(),
                                                      input->get_tensor()->get_bytes(),
                                                      event);
            }
        }

        this->_builder->set_input_list(this->_inputs);
        this->_builder->set_name(this->_node_metadata._name);
        this->_builder->set_mem_manager(this->_mem_manager_ptr);
        this->_builder->set_event(this->event());
        this->_builder->set_wait_list(events_list);
        this->_builder->set_stream(this->stream());


        auto callback = kernel::base::get_op_func(this->_tensor->get_allocator(),
                                                  this->_op_type, this->data_type());
        return callback;
    };
    //
    void GraphNode::add_input_node(const std::shared_ptr<GraphNode> &src_node) {
        if (src_node == nullptr) {
            return ;
        }
        auto iter = std::find(this->_input_nodes.begin(), this->_input_nodes.end(), src_node);
        if (iter == this->_input_nodes.end()) {
            for (auto &node: this->input_nodes()) {
                if (node->name() == src_node->name()) {
                    return ;
                }
            }
            this->set_layer_id(src_node->layer_id());
            this->_input_nodes.push_back(src_node);
        }
    }
}
