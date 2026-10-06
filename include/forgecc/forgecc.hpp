#pragma once
#include "forgecc/forgecc.h"
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <memory>
#include <iostream>

namespace forge {

enum class Device {
    CPU = FORGE_DEV_CPU,
    GPU = FORGE_DEV_GPU
};

class Tensor {
public:
    Tensor(std::vector<int64_t> shape, Device dev = Device::CPU)
        : raw_(forge_tensor_create(shape.data(), static_cast<int>(shape.size()), static_cast<ForgeDevice>(dev))),
          shape_(std::move(shape)),
          dev_(dev) {
        if (!raw_) {
            throw std::runtime_error("Failed to allocate ForgeTensor.");
        }
    }

    Tensor(std::initializer_list<int64_t> shape, Device dev = Device::CPU)
        : Tensor(std::vector<int64_t>(shape), dev) {}

    ~Tensor() {
        if (raw_) {
            forge_tensor_free(raw_);
            raw_ = nullptr;
        }
    }

    Tensor(const Tensor& other)
        : Tensor(other.shape_, other.dev_) {
        std::vector<float> buf(num_elements());
        other.to_host(buf.data());
        from_host(buf.data());
    }

    Tensor(Tensor&& other) noexcept
        : raw_(other.raw_), shape_(std::move(other.shape_)), dev_(other.dev_) {
        other.raw_ = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (raw_) forge_tensor_free(raw_);
            raw_ = other.raw_;
            shape_ = std::move(other.shape_);
            dev_ = other.dev_;
            other.raw_ = nullptr;
        }
        return *this;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (raw_) forge_tensor_free(raw_);
            shape_ = other.shape_;
            dev_ = other.dev_;
            raw_ = forge_tensor_create(shape_.data(), static_cast<int>(shape_.size()), static_cast<ForgeDevice>(dev_));
            std::vector<float> buf(num_elements());
            other.to_host(buf.data());
            from_host(buf.data());
        }
        return *this;
    }

    int64_t num_elements() const {
        return forge_tensor_num_elements(raw_);
    }

    const std::vector<int64_t>& shape() const {
        return shape_;
    }

    Device device() const {
        return dev_;
    }

    float* data() {
        return forge_tensor_data(raw_);
    }

    const float* data() const {
        return forge_tensor_data_const(raw_);
    }

    void from_host(const float* host_src) {
        forge_tensor_copy_from_host(raw_, host_src);
    }

    void to_host(float* host_dst) const {
        forge_tensor_copy_to_host(raw_, host_dst);
    }

    Tensor to(Device target_dev) const {
        if (dev_ == target_dev) {
            return *this;
        }
        Tensor res(shape_, target_dev);
        std::vector<float> host_buf(num_elements());
        to_host(host_buf.data());
        res.from_host(host_buf.data());
        return res;
    }

    ForgeTensor* raw() { return raw_; }
    const ForgeTensor* raw() const { return raw_; }

private:
    ForgeTensor* raw_ = nullptr;
    std::vector<int64_t> shape_;
    Device dev_;
};

inline bool has_gpu() {
    return forge_has_gpu_support();
}

inline void sync() {
    forge_sync();
}

inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2) {
        throw std::invalid_argument("MatMul requires 2D tensors.");
    }
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    if (B.shape()[0] != K) {
        throw std::invalid_argument("Dimension mismatch in matmul.");
    }
    int64_t N = B.shape()[1];

    Tensor C({M, N}, A.device());
    forge_op_matmul(A.raw(), B.raw(), C.raw());
    return C;
}

inline Tensor relu(const Tensor& in) {
    Tensor out(in.shape(), in.device());
    forge_op_relu(in.raw(), out.raw());
    return out;
}

inline Tensor matmul_relu(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2) {
        throw std::invalid_argument("MatMul requires 2D tensors.");
    }
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    if (B.shape()[0] != K) {
        throw std::invalid_argument("Dimension mismatch in matmul.");
    }
    int64_t N = B.shape()[1];

    Tensor C({M, N}, A.device());
    forge_op_matmul_relu(A.raw(), B.raw(), C.raw());
    return C;
}

}
