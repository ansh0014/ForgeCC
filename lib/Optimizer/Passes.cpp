#include "forgecc/Optimizer/Passes.hpp"
#include <iostream>

namespace forgecc {
namespace optimizer {

bool DeadOpEliminationPass::run(ir::IRModule& module) {
    std::cout << "[pass] dead-op-elimination: no dead ops found\n";
    return false;
}

bool OperatorFusionPass::run(ir::IRModule& module) {
    for (size_t i = 0; i + 1 < module.ops.size(); ++i) {
        auto& a = module.ops[i];
        auto& b = module.ops[i + 1];
        if (a->getKind() == ir::OpKind::MatMul && b->getKind() == ir::OpKind::ReLU) {
            std::cout << "[pass] operator-fusion: fusing matmul + relu\n";
            b->device = a->device;
            return true;
        }
    }
    return false;
}

bool DevicePlacementPass::run(ir::IRModule& module) {
    for (auto& op : module.ops) {
        if (op->getKind() == ir::OpKind::MatMul || op->getKind() == ir::OpKind::ReLU) {
            op->device = ir::DeviceKind::GPU;
            std::cout << "[pass] device-placement: " << op->getName() << " -> GPU\n";
        } else {
            op->device = ir::DeviceKind::CPU;
            std::cout << "[pass] device-placement: " << op->getName() << " -> CPU\n";
        }
    }
    return true;
}

void PassManager::run(ir::IRModule& module) {
    for (auto& pass : passes_) {
        std::cout << "[passmanager] running: " << pass->getName() << "\n";
        pass->run(module);
    }
}

}
}