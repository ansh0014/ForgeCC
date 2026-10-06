#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FORGE_DEV_CPU = 0,
    FORGE_DEV_GPU = 1
} ForgeDevice;

typedef struct ForgeTensor ForgeTensor;

ForgeTensor* forge_tensor_create(const int64_t* shape, int ndim, ForgeDevice dev);
void         forge_tensor_free(ForgeTensor* t);

int64_t      forge_tensor_ndim(const ForgeTensor* t);
int64_t      forge_tensor_dim(const ForgeTensor* t, int axis);
int64_t      forge_tensor_num_elements(const ForgeTensor* t);
ForgeDevice  forge_tensor_device(const ForgeTensor* t);
float*       forge_tensor_data(ForgeTensor* t);
const float* forge_tensor_data_const(const ForgeTensor* t);

void         forge_tensor_copy_from_host(ForgeTensor* dst, const float* src_host);
void         forge_tensor_copy_to_host(const ForgeTensor* src, float* dst_host);
ForgeTensor* forge_tensor_to_device(const ForgeTensor* src, ForgeDevice target_dev);

void         forge_op_matmul(const ForgeTensor* A, const ForgeTensor* B, ForgeTensor* C);
void         forge_op_relu(const ForgeTensor* in, ForgeTensor* out);
void         forge_op_matmul_relu(const ForgeTensor* A, const ForgeTensor* B, ForgeTensor* C);

bool         forge_has_gpu_support(void);
void         forge_sync(void);

#ifdef __cplusplus
}
#endif
