#include "forgecc/forgecc.h"
#include "forgecc_rt/gpu_runtime.hpp"
#include <vector>
#include <cstdlib>
#include <cstring>
#include <new>

extern "C" {
void cpu_matmul(const float* A, const float* B, float* C,
                int64_t M, int64_t K, int64_t N);
void cpu_relu(const float* in, float* out, int64_t n);
}

struct ForgeTensor {
    std::vector<int64_t> shape;
    int64_t num_elements;
    ForgeDevice dev;
    void* ptr;
};

extern "C" {

ForgeTensor* forge_tensor_create(const int64_t* shape, int ndim, ForgeDevice dev) {
    if (!shape || ndim <= 0) return nullptr;

    ForgeTensor* t = new (std::nothrow) ForgeTensor();
    if (!t) return nullptr;

    t->shape.assign(shape, shape + ndim);
    t->ndim = ndim;
    t->dev = dev;

    int64_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        total *= shape[i];
    }
    t->num_elements = total;

    size_t bytes = static_cast<size_t>(total) * sizeof(float);
    if (dev == FORGE_DEV_GPU) {
        t->ptr = forge_gpu_malloc(bytes);
    } else {
        t->ptr = std::malloc(bytes);
        if (t->ptr) {
            std::memset(t->ptr, 0, bytes);
        }
    }

    if (!t->ptr) {
        delete t;
        return nullptr;
    }

    return t;
}

void forge_tensor_free(ForgeTensor* t) {
    if (!t) return;
    if (t->ptr) {
        if (t->dev == FORGE_DEV_GPU) {
            forge_gpu_free(t->ptr);
        } else {
            std::free(t->ptr);
        }
    }
    delete t;
}

int64_t forge_tensor_ndim(const ForgeTensor* t) {
    return t ? static_cast<int64_t>(t->shape.size()) : 0;
}

int64_t forge_tensor_dim(const ForgeTensor* t, int axis) {
    if (!t || axis < 0 || axis >= static_cast<int>(t->shape.size())) return 0;
    return t->shape[axis];
}

int64_t forge_tensor_num_elements(const ForgeTensor* t) {
    return t ? t->num_elements : 0;
}

ForgeDevice forge_tensor_device(const ForgeTensor* t) {
    return t ? t->dev : FORGE_DEV_CPU;
}

float* forge_tensor_data(ForgeTensor* t) {
    return t ? reinterpret_cast<float*>(t->ptr) : nullptr;
}

const float* forge_tensor_data_const(const ForgeTensor* t) {
    return t ? reinterpret_cast<const float*>(t->ptr) : nullptr;
}

void forge_tensor_copy_from_host(ForgeTensor* dst, const float* src_host) {
    if (!dst || !src_host) return;
    size_t bytes = static_cast<size_t>(dst->num_elements) * sizeof(float);
    if (dst->dev == FORGE_DEV_GPU) {
        forge_gpu_memcpy_to_device(dst->ptr, src_host, bytes);
    } else {
        std::memcpy(dst->ptr, src_host, bytes);
    }
}

void forge_tensor_copy_to_host(const ForgeTensor* src, float* dst_host) {
    if (!src || !dst_host) return;
    size_t bytes = static_cast<size_t>(src->num_elements) * sizeof(float);
    if (src->dev == FORGE_DEV_GPU) {
        forge_gpu_memcpy_to_host(dst_host, src->ptr, bytes);
    } else {
        std::memcpy(dst_host, src->ptr, bytes);
    }
}

ForgeTensor* forge_tensor_to_device(const ForgeTensor* src, ForgeDevice target_dev) {
    if (!src) return nullptr;
    ForgeTensor* dst = forge_tensor_create(src->shape.data(), static_cast<int>(src->shape.size()), target_dev);
    if (!dst) return nullptr;

    size_t bytes = static_cast<size_t>(src->num_elements) * sizeof(float);
    if (src->dev == target_dev) {
        if (target_dev == FORGE_DEV_GPU) {
            std::vector<float> tmp(src->num_elements);
            forge_gpu_memcpy_to_host(tmp.data(), src->ptr, bytes);
            forge_gpu_memcpy_to_device(dst->ptr, tmp.data(), bytes);
        } else {
            std::memcpy(dst->ptr, src->ptr, bytes);
        }
    } else if (src->dev == FORGE_DEV_CPU && target_dev == FORGE_DEV_GPU) {
        forge_gpu_memcpy_to_device(dst->ptr, src->ptr, bytes);
    } else if (src->dev == FORGE_DEV_GPU && target_dev == FORGE_DEV_CPU) {
        forge_gpu_memcpy_to_host(dst->ptr, src->ptr, bytes);
    }

    return dst;
}

void forge_op_matmul(const ForgeTensor* A, const ForgeTensor* B, ForgeTensor* C) {
    if (!A || !B || !C) return;
    int64_t M = A->shape[0];
    int64_t K = A->shape[1];
    int64_t N = B->shape[1];

    if (A->dev == FORGE_DEV_GPU && B->dev == FORGE_DEV_GPU && C->dev == FORGE_DEV_GPU) {
        forge_gpu_matmul(reinterpret_cast<const float*>(A->ptr),
                         reinterpret_cast<const float*>(B->ptr),
                         reinterpret_cast<float*>(C->ptr),
                         M, K, N);
    } else {
        cpu_matmul(reinterpret_cast<const float*>(A->ptr),
                   reinterpret_cast<const float*>(B->ptr),
                   reinterpret_cast<float*>(C->ptr),
                   M, K, N);
    }
}

void forge_op_relu(const ForgeTensor* in, ForgeTensor* out) {
    if (!in || !out) return;
    int64_t n = in->num_elements;

    if (in->dev == FORGE_DEV_GPU && out->dev == FORGE_DEV_GPU) {
        forge_gpu_relu(reinterpret_cast<const float*>(in->ptr),
                       reinterpret_cast<float*>(out->ptr),
                       n);
    } else {
        cpu_relu(reinterpret_cast<const float*>(in->ptr),
                 reinterpret_cast<float*>(out->ptr),
                 n);
    }
}

void forge_op_matmul_relu(const ForgeTensor* A, const ForgeTensor* B, ForgeTensor* C) {
    if (!A || !B || !C) return;
    int64_t M = A->shape[0];
    int64_t K = A->shape[1];
    int64_t N = B->shape[1];

    if (A->dev == FORGE_DEV_GPU && B->dev == FORGE_DEV_GPU && C->dev == FORGE_DEV_GPU) {
        forge_gpu_matmul_relu_fused(reinterpret_cast<const float*>(A->ptr),
                                    reinterpret_cast<const float*>(B->ptr),
                                    reinterpret_cast<float*>(C->ptr),
                                    M, K, N);
    } else {
        cpu_matmul(reinterpret_cast<const float*>(A->ptr),
                   reinterpret_cast<const float*>(B->ptr),
                   reinterpret_cast<float*>(C->ptr),
                   M, K, N);
        cpu_relu(reinterpret_cast<const float*>(C->ptr),
                 reinterpret_cast<float*>(C->ptr),
                 M * N);
    }
}

bool forge_has_gpu_support(void) {
    return forge_gpu_is_available();
}

void forge_sync(void) {
    forge_gpu_sync();
}

}
