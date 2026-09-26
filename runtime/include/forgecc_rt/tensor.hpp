#pragma once
#include <vector>
#include <cstdint>

namespace forgecc {

template <typename T>
class Tensor {
public:
    explicit Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
        int64_t size = 1;
        for (auto d : shape_) size *= d;
        data_.resize(size, T{});
    }

    const std::vector<int64_t>& shape() const { return shape_; }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    int64_t numElements() const { return (int64_t)data_.size(); }

private:
    std::vector<int64_t> shape_;
    std::vector<T>       data_;
};

}

