#include "Driver.hpp"
#include "forgecc/AST/ASTNodes.hpp"
#include "forgecc/Sema/Sema.hpp"
#include "forgecc/IR/TensorIR.hpp"
#include "forgecc/Optimizer/Passes.hpp"
#include "forgecc/CodeGen/CodeGen.hpp"
#include "forgecc/Support/Diagnostics.hpp"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <unordered_map>

static std::unique_ptr<forgecc::ast::ProgramNode> buildDemoAST() {
    auto prog = std::make_unique<forgecc::ast::ProgramNode>();
    prog->statements.push_back(
        std::make_unique<forgecc::ast::TensorDeclNode>("A", "f32", std::vector<int64_t>{1024, 1024}));
    prog->statements.push_back(
        std::make_unique<forgecc::ast::TensorDeclNode>("B", "f32", std::vector<int64_t>{1024, 1024}));
    prog->statements.push_back(
        std::make_unique<forgecc::ast::MatMulNode>("A", "B", "C"));
    prog->statements.push_back(
        std::make_unique<forgecc::ast::ReLUNode>("C", "D"));
    return prog;
}

static forgecc::ir::IRModule buildIRFromAST(const forgecc::ast::ProgramNode& prog) {
    forgecc::ir::IRModule module("forgecc_module");
    std::unordered_map<std::string, forgecc::ir::Value*> valueMap;

    for (const auto& stmt : prog.statements) {
        using namespace forgecc::ast;

        if (stmt->getKind() == ASTKind::TensorDecl) {
            auto* decl = static_cast<TensorDeclNode*>(stmt.get());
            auto val = std::make_unique<forgecc::ir::Value>(
                decl->name,
                forgecc::ir::TensorType{decl->dataType, decl->shape});
            valueMap[decl->name] = val.get();
            module.addValue(std::move(val));

        } else if (stmt->getKind() == ASTKind::MatMulOp) {
            auto* mm = static_cast<MatMulNode*>(stmt.get());
            auto out = std::make_unique<forgecc::ir::Value>(
                mm->output,
                forgecc::ir::TensorType{"f32", {1024, 1024}});
            auto* outPtr = out.get();
            module.addValue(std::move(out));

            auto op = std::make_unique<forgecc::ir::MatMulOp>();
            op->operands = {valueMap[mm->lhs], valueMap[mm->rhs]};
            op->result   = outPtr;
            valueMap[mm->output] = outPtr;
            module.addOp(std::move(op));

        } else if (stmt->getKind() == ASTKind::ReLUOp) {
            auto* ru = static_cast<ReLUNode*>(stmt.get());
            auto out = std::make_unique<forgecc::ir::Value>(
                ru->output,
                forgecc::ir::TensorType{"f32", {1024, 1024}});
            auto* outPtr = out.get();
            module.addValue(std::move(out));

            auto op = std::make_unique<forgecc::ir::ReLUOp>();
            op->operands = {valueMap[ru->input]};
            op->result   = outPtr;
            valueMap[ru->output] = outPtr;
            module.addOp(std::move(op));
        }
    }
    return module;
}

void Driver::parseArgs(int argc, char* argv[]) {
    inputFile    = argv[1];
    outputFile   = "output";
    emitIR       = false;
    emitForgeIR  = false;
    emitObj      = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--emit-ir")       emitIR = true;
        else if (arg == "--emit-forge-ir") emitForgeIR = true;
        else if (arg == "--emit-obj")      emitObj = true;
        else if (arg == "-o" && i + 1 < argc) outputFile = argv[++i];
    }
}

void Driver::compile() {
    forgecc::DiagnosticEngine diags;

    std::cout << "[1/4] Building AST...\n";
    auto prog = buildDemoAST();

    std::cout << "[2/4] Running Semantic Analysis...\n";
    forgecc::SemanticAnalyzer sema(diags);
    if (!sema.analyze(*prog)) return;

    std::cout << "[3/4] Lowering to ForgeCC IR...\n";
    auto irModule = buildIRFromAST(*prog);

    if (emitForgeIR) {
        irModule.dump();
        return;
    }

    forgecc::optimizer::PassManager pm;
    pm.addPass(std::make_unique<forgecc::optimizer::DevicePlacementPass>());
    pm.addPass(std::make_unique<forgecc::optimizer::OperatorFusionPass>());
    pm.addPass(std::make_unique<forgecc::optimizer::DeadOpEliminationPass>());
    pm.run(irModule);

    std::cout << "[4/4] Generating LLVM IR...\n";
    forgecc::codegen::LLVMCodeGen cg(diags, "forgecc_module");
    cg.generate(irModule);

    if (emitIR)       cg.emitIR();
    else if (emitObj) cg.emitObjectFile(outputFile + ".o");
    else              std::cout << "forgecc: compiled " << inputFile << " -> " << outputFile << "\n";
}

int Driver::run(int argc, char* argv[]) {
    parseArgs(argc, argv);
    std::cout << "forgecc: processing " << inputFile << "\n";
    compile();
    return 0;
}