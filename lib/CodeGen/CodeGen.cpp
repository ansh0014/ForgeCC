#include "forgecc/CodeGen/CodeGen.hpp"
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <unordered_map>
#include <iostream>

namespace forgecc {
namespace codegen {

LLVMCodeGen::LLVMCodeGen(DiagnosticEngine& diags, const std::string& moduleName)
    : diags_(diags),
      module_(std::make_unique<llvm::Module>(moduleName, ctx_)),
      builder_(ctx_) {}

bool LLVMCodeGen::generate(const ir::IRModule& irModule) {
    auto* ptrTy  = llvm::PointerType::get(ctx_, 0);
    auto* i64Ty  = llvm::Type::getInt64Ty(ctx_);
    auto* voidTy = llvm::Type::getVoidTy(ctx_);

    auto mallocFn = module_->getOrInsertFunction("malloc",
        llvm::FunctionType::get(ptrTy, {i64Ty}, false));

    auto freeFn = module_->getOrInsertFunction("free",
        llvm::FunctionType::get(voidTy, {ptrTy}, false));

    auto matmulFn = module_->getOrInsertFunction("cpu_matmul",
        llvm::FunctionType::get(voidTy,
            {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty}, false));

    auto reluFn = module_->getOrInsertFunction("cpu_relu",
        llvm::FunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty}, false));

    auto* funcType = llvm::FunctionType::get(builder_.getInt32Ty(), false);
    auto* func = llvm::Function::Create(
        funcType, llvm::Function::ExternalLinkage, "main", module_.get());
    auto* entry = llvm::BasicBlock::Create(ctx_, "entry", func);
    builder_.SetInsertPoint(entry);

    std::unordered_map<std::string, llvm::Value*> tensorPtrs;

    for (const auto& val : irModule.values) {
        int64_t numElements = 1;
        for (auto d : val->type.shape) numElements *= d;
        auto* bytes = builder_.getInt64(numElements * sizeof(float));
        auto* ptr   = builder_.CreateCall(mallocFn, {bytes}, val->name + "_ptr");
        tensorPtrs[val->name] = ptr;
    }

    for (const auto& op : irModule.ops) {
        if (op->getKind() == ir::OpKind::MatMul
            && op->operands.size() == 2
            && op->result) {
            auto* A = tensorPtrs[op->operands[0]->name];
            auto* B = tensorPtrs[op->operands[1]->name];
            auto* C = tensorPtrs[op->result->name];
            auto* M = builder_.getInt64(op->result->type.shape[0]);
            auto* K = builder_.getInt64(op->operands[0]->type.shape[1]);
            auto* N = builder_.getInt64(op->result->type.shape[1]);
            builder_.CreateCall(matmulFn, {A, B, C, M, K, N});

        } else if (op->getKind() == ir::OpKind::ReLU
                   && op->operands.size() == 1
                   && op->result) {
            auto* in  = tensorPtrs[op->operands[0]->name];
            auto* out = tensorPtrs[op->result->name];
            int64_t n = 1;
            for (auto d : op->result->type.shape) n *= d;
            builder_.CreateCall(reluFn, {in, out, builder_.getInt64(n)});
        }
    }

    for (const auto& val : irModule.values) {
        builder_.CreateCall(freeFn, {tensorPtrs[val->name]});
    }

    builder_.CreateRet(builder_.getInt32(0));
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
