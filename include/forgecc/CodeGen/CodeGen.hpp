#pragma once
#include "forgecc/IR/TensorIR.hpp"
#include "forgecc/Support/Diagnostics.hpp"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <memory>
#include <string>

namespace forgecc {
namespace codegen {

class LLVMCodeGen {
public:
    LLVMCodeGen(DiagnosticEngine& diags, const std::string& moduleName);

    bool generate(const ir::IRModule& irModule);
    void emitIR();
    void emitObjectFile(const std::string& path);

private:
    DiagnosticEngine&             diags_;
    llvm::LLVMContext             ctx_;
    std::unique_ptr<llvm::Module> module_;
    llvm::IRBuilder<>             builder_;

    void buildMainWrapper();
};

}
}
