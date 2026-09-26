#include "forgecc/IR/TensorIR.hpp"
#include <iostream>

namespace forgecc {
namespace ir {

void IRModule::dump() const {
    std::cout << "; ForgeCC IR Module: " << name << "\n\n";

    for (const auto& val : values) {
        std::cout << "tensor " << val->name << " : " << val->type.dtype << " [";
        for (size_t i = 0; i < val->type.shape.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << val->type.shape[i];
        }
        std::cout << "]\n";
    }

    std::cout << "\n";

    for (const auto& op : ops) {
        std::cout << "  " << op->getName() << " : ";
        for (auto* operand : op->operands)
            std::cout << operand->name << " ";
        if (op->result)
            std::cout << "-> " << op->result->name;
        std::cout << "\n";
    }
}

}
}