#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace forgecc {
namespace ast {

enum class ASTKind { Program, TensorDecl, MatMulOp, ReLUOp, AddOp };

class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual ASTKind getKind() const = 0;
};


class TensorDeclNode : public ASTNode {
public:
    std::string name;
    std::string dataType;
    std::vector<int64_t> shape;

    TensorDeclNode(std::string name, std::string dtype, std::vector<int64_t> shape)
        : name(std::move(name)), dataType(std::move(dtype)), shape(std::move(shape)) {}

    ASTKind getKind() const override { return ASTKind::TensorDecl; }
};

class MatMulNode : public ASTNode {
public:
    std::string lhs, rhs, output;

    MatMulNode(std::string lhs, std::string rhs, std::string out)
        : lhs(std::move(lhs)), rhs(std::move(rhs)), output(std::move(out)) {}

    ASTKind getKind() const override { return ASTKind::MatMulOp; }
};

// ---------- ReLU: auto D = relu(C); ----------
class ReLUNode : public ASTNode {
public:
    std::string input, output;

    ReLUNode(std::string in, std::string out)
        : input(std::move(in)), output(std::move(out)) {}

    ASTKind getKind() const override { return ASTKind::ReLUOp; }
};


class AddNode : public ASTNode {
public:
    std::string lhs, rhs, output;

    AddNode(std::string lhs, std::string rhs, std::string out)
        : lhs(std::move(lhs)), rhs(std::move(rhs)), output(std::move(out)) {}

    ASTKind getKind() const override { return ASTKind::AddOp; }
};


class ProgramNode : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> statements;
    ASTKind getKind() const override { return ASTKind::Program; }
};

} // namespace ast
} // namespace forgecc