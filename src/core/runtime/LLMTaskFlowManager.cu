//
// Created by nkk on 2025/11/10.
//

#include "LLMTaskFlowManager.h"

#include "global/ModelGlobalVar.h"

namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(LLMTaskFlowManager, tff::module::ModuleObject, TASK_FLOW_MANAGER_FLAG,tff::core::global::TaskFlowType::TFF_FLOW_LLM);
    void LLMTaskFlowManager::build_task_flow(const std::shared_ptr<tff::core::graph::Graph> &_graph_ptr) {

    }
}
