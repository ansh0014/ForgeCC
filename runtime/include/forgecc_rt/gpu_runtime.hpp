#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

bool forge_gpu_is_available();
void forge_gpu_init();

void* forge_gpu_malloc(size_t bytes);
void  forge_gpu_free(void* ptr);

void forge_gpu_memcpy_to_device(void* dst, const void* src, size_t bytes);
void forge_gpu_memcpy_to_host(void* dst, const void* src, size_t bytes);

void forge_gpu_matmul(const float* A, const float* B, float* C,
                      int64_t M, int64_t K, int64_t N);

void forge_gpu_relu(const float* in, float* out, int64_t n);

void forge_gpu_matmul_relu_fused(const float* A, const float* B, float* C,
                                 int64_t M, int64_t K, int64_t N);

void forge_gpu_sync();

#ifdef __cplusplus
}
#endif
