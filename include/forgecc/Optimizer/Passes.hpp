#pragma once
#include "forgecc/IR/TensorIR.hpp"
#include <string>
#include <vector>
#include <memory>

namespace forgecc {
namespace optimizer {

class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string getName() const = 0;
    virtual bool run(ir::IRModule& module) = 0;
};

class DeadOpEliminationPass : public Pass {
public:
    std::string getName() const override { return "dead-op-elimination"; }
    bool run(ir::IRModule& module) override;
};

class OperatorFusionPass : public Pass {
public:
    std::string getName() const override { return "operator-fusion"; }
    bool run(ir::IRModule& module) override;
};

class DevicePlacementPass : public Pass {
public:
    std::string getName() const override { return "device-placement"; }
    bool run(ir::IRModule& module) override;
};

class PassManager {
public:
    void addPass(std::unique_ptr<Pass> pass) {
        passes_.push_back(std::move(pass));
    }
    void run(ir::IRModule& module);

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

}
}
