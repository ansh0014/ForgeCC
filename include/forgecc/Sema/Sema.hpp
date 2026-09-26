#pragma once
#include "forgecc/AST/ASTNodes.hpp"
#include "forgecc/Support/Diagnostics.hpp"
#include <unordered_map>
#include <string>

namespace forgecc {

struct TensorSymbol {
    std::string name;
    std::string dataType;
    std::vector<int64_t> shape;
};

class SemanticAnalyzer {
private:
    DiagnosticEngine& diags;
    std::unordered_map<std::string, TensorSymbol> symbolTable;

public:
    explicit SemanticAnalyzer(DiagnosticEngine& diags) : diags(diags) {}

    bool analyze(const ast::ProgramNode& program) {
        for (const auto& stmt : program.statements) {
            if (stmt->getKind() == ast::ASTKind::TensorDecl) {
                auto* decl = static_cast<ast::TensorDeclNode*>(stmt.get());
                symbolTable[decl->name] = {decl->name, decl->dataType, decl->shape};
            } else if (stmt->getKind() == ast::ASTKind::MatMulOp) {
                auto* mm = static_cast<ast::MatMulNode*>(stmt.get());
                if (!symbolTable.count(mm->lhs)) {
                    diags.report(DiagLevel::Error, "Undeclared tensor: " + mm->lhs);
                    return false;
                }
                if (!symbolTable.count(mm->rhs)) {
                    diags.report(DiagLevel::Error, "Undeclared tensor: " + mm->rhs);
                    return false;
                }
            } else if (stmt->getKind() == ast::ASTKind::ReLUOp) {
                auto* ru = static_cast<ast::ReLUNode*>(stmt.get());
                if (!symbolTable.count(ru->input)) {
                    diags.report(DiagLevel::Error, "Undeclared tensor: " + ru->input);
                    return false;
                }
            }
        }
        return true;
    }
};

}
