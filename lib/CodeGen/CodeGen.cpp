#include "forgecc/CodeGen/CodeGen.hpp"
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/LegacyPassManager.h>
#include <iostream>

namespace forgecc {
namespace codegen {

LLVMCodeGen::LLVMCodeGen(DiagnosticEngine& diags, const std::string& moduleName)
    : diags_(diags),
      module_(std::make_unique<llvm::Module>(moduleName, ctx_)),
      builder_(ctx_) {}

void LLVMCodeGen::buildMainWrapper() {
    auto* funcType = llvm::FunctionType::get(builder_.getInt32Ty(), false);
    auto* func = llvm::Function::Create(
        funcType, llvm::Function::ExternalLinkage, "main", module_.get());
    auto* entry = llvm::BasicBlock::Create(ctx_, "entry", func);
    builder_.SetInsertPoint(entry);
    builder_.CreateRet(builder_.getInt32(0));
}

bool LLVMCodeGen::generate(const ir::IRModule& irModule) {
    buildMainWrapper();
    return true;
}

void LLVMCodeGen::emitIR() {
    module_->print(llvm::outs(), nullptr);
}

void LLVMCodeGen::emitObjectFile(const std::string& path) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
    module_->setTargetTriple(triple);

    std::string error;
    auto* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        diags_.report(DiagLevel::Error, "Target lookup failed: " + error);
        return;
    }

    llvm::TargetOptions opt;
    auto* machine = target->createTargetMachine(triple, "generic", "", opt, std::nullopt);
    module_->setDataLayout(machine->createDataLayout());

    std::error_code ec;
    llvm::raw_fd_ostream dest(path, ec, llvm::sys::fs::OF_None);
    if (ec) {
        diags_.report(DiagLevel::Error, "Cannot open output file: " + ec.message());
        return;
    }

    llvm::legacy::PassManager pm;
    if (machine->addPassesToEmitFile(pm, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        diags_.report(DiagLevel::Error, "Cannot emit object file for target");
        return;
    }
    pm.run(*module_);
    dest.flush();
}

}
}
