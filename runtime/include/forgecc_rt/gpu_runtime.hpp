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

void forge_gpu_matmul_add_fused(const float* A, const float* B, const float* bias, float* C,
                                int64_t M, int64_t K, int64_t N);

void forge_gpu_matmul_add_relu_fused(const float* A, const float* B, const float* bias, float* C,
                                     int64_t M, int64_t K, int64_t N);

void forge_gpu_matmul_add_relu_residual_fused(const float* A, const float* B, const float* bias, const float* residual, float* C,
                                              int64_t M, int64_t K, int64_t N);

void forge_gpu_add_relu_fused(const float* a, const float* b, float* out, int64_t n);

bool forge_gpu_get_device_name(char* out_name, int max_len);
int  forge_gpu_get_attribute(int attribute);
bool forge_gpu_get_memory_info(size_t* free_bytes, size_t* total_bytes);

void forge_gpu_auto_tune_elementwise(int64_t n, unsigned int* block_size, unsigned int* grid_size);
void forge_gpu_auto_tune_matmul(int64_t M, int64_t N, unsigned int* block_x, unsigned int* block_y, unsigned int* grid_x, unsigned int* grid_y);

void forge_gpu_relu_backward(const float* grad_out, const float* in, float* grad_in, int64_t n);
void forge_gpu_sgd_step(float* weights, const float* grads, float* velocity, float lr, float momentum, float weight_decay, int64_t n);
void forge_gpu_adam_step(float* weights, const float* grads, float* m, float* v, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, int64_t n);

void forge_gpu_sync();

#ifdef __cplusplus
}
#endif
