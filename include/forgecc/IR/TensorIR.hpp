#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace forgecc {
namespace ir {

enum class DeviceKind { Unknown, CPU, GPU };
enum class OpKind     { TensorLoad, MatMul, ReLU, Add };

struct TensorType {
    std::string      dtype;
    std::vector<int64_t> shape;
    DeviceKind       device = DeviceKind::Unknown;
};

class Value {
public:
    std::string name;
    TensorType  type;

    Value(std::string name, TensorType type)
        : name(std::move(name)), type(std::move(type)) {}
};

class Operation {
public:
    virtual ~Operation() = default;
    virtual OpKind      getKind() const = 0;
    virtual std::string getName() const = 0;

    std::vector<Value*> operands;
    Value*              result = nullptr;
    DeviceKind          device = DeviceKind::Unknown;
};

class MatMulOp : public Operation {
public:
    OpKind getKind() const override { return OpKind::MatMul; }
    std::string getName() const override { return "matmul"; }
};

class ReLUOp : public Operation {
public:
    OpKind getKind() const override { return OpKind::ReLU; }
    std::string getName() const override { return "relu"; }
};

class AddOp : public Operation {
public:
    OpKind getKind() const override { return OpKind::Add; }
    std::string getName() const override { return "add"; }
};

class IRModule {
public:
    std::string name;
    std::vector<std::unique_ptr<Value>>     values;
    std::vector<std::unique_ptr<Operation>> ops;

    IRModule(std::string name) : name(std::move(name)) {}

    void addValue(std::unique_ptr<Value> v)     { values.push_back(std::move(v)); }
    void addOp(std::unique_ptr<Operation> op)   { ops.push_back(std::move(op)); }

    void dump() const;
};

}
}

