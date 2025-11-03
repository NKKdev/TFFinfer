//
// Created by nkk on 2025/11/3.
//

#include "TFFOPNode.h"
namespace tff::core::graph::op {
    REGISTER_MODULE_OBJECT(MatMulNode, GraphNode, OP_NODE_FLAG, MAT_MUL_NODE, MAT_MUL_NODE);
}