#include "Driver.hpp"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>

#include <iostream>
#include <string>

void Driver::parseArgs(int argc, char* argv[]) {
    inputFile = argv[1];
    outputFile = "output";
    emitIR = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--emit-ir") {
            emitIR = true;
        } else if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        }
    }
}

void Driver::compile() {
    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("forgecc_module", context);
    llvm::IRBuilder<> builder(context);


    llvm::FunctionType* funcType = llvm::FunctionType::get(builder.getInt32Ty(), false);
    llvm::Function* mainFunc = llvm::Function::Create(
        funcType, llvm::Function::ExternalLinkage, "main", module.get());

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", mainFunc);
    builder.SetInsertPoint(entry);
    builder.CreateRet(builder.getInt32(0));

    if (emitIR) {
        module->print(llvm::outs(), nullptr);
    } else {
        std::cout << "forgecc: compiled " << inputFile << " -> " << outputFile << "\n";
    }
}

int Driver::run(int argc, char* argv[]) {
    parseArgs(argc, argv);
    std::cout << "forgecc: processing " << inputFile << "\n";
    compile();
    return 0;
}