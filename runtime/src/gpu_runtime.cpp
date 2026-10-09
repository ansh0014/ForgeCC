#include "forgecc_rt/gpu_runtime.hpp"
#include "forgecc_rt/tensor.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

typedef CUresult (*PFN_cuInit)(unsigned int);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*PFN_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (*PFN_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*PFN_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*PFN_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, void*, void**, void**);
typedef CUresult (*PFN_cuCtxSynchronize)(void);
typedef CUresult (*PFN_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (*PFN_cuDeviceGetAttribute)(int*, int, CUdevice);
typedef CUresult (*PFN_cuMemGetInfo)(size_t*, size_t*);

struct CudaDriverAPI {
    PFN_cuInit              cuInit = nullptr;
    PFN_cuDeviceGet         cuDeviceGet = nullptr;
    PFN_cuCtxCreate         cuCtxCreate = nullptr;
    PFN_cuMemAlloc          cuMemAlloc = nullptr;
    PFN_cuMemFree           cuMemFree = nullptr;
    PFN_cuMemcpyHtoD        cuMemcpyHtoD = nullptr;
    PFN_cuMemcpyDtoH        cuMemcpyDtoH = nullptr;
    PFN_cuModuleLoadData    cuModuleLoadData = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuLaunchKernel      cuLaunchKernel = nullptr;
    PFN_cuCtxSynchronize    cuCtxSynchronize = nullptr;
    PFN_cuDeviceGetName     cuDeviceGetName = nullptr;
    PFN_cuDeviceGetAttribute cuDeviceGetAttribute = nullptr;
    PFN_cuMemGetInfo        cuMemGetInfo = nullptr;

    bool initialized = false;
    bool available = false;
    CUcontext context = nullptr;
    CUmodule  module = nullptr;

    CUfunction fn_matmul = nullptr;
    CUfunction fn_relu = nullptr;
    CUfunction fn_add = nullptr;
    CUfunction fn_sub = nullptr;
    CUfunction fn_mul = nullptr;
    CUfunction fn_matmul_relu = nullptr;
    CUfunction fn_matmul_add = nullptr;
    CUfunction fn_matmul_add_relu = nullptr;
    CUfunction fn_matmul_add_relu_residual = nullptr;
    CUfunction fn_add_relu = nullptr;
};

static CudaDriverAPI g_cuda;

static const char* kPtxSource = 
".version 7.0\n"
".target sm_50\n"
".address_size 64\n"
"\n"
".visible .entry gpu_matmul_kernel(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .s64 param_M,\n"
"    .param .s64 param_K,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<30>;\n"
"    .reg .f32 %f<10>;\n"
"    .reg .pred %p<10>;\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_C];\n"
"    ld.param.s64 %rd4, [param_M];\n"
"    ld.param.s64 %rd5, [param_K];\n"
"    ld.param.s64 %rd6, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd7, %r4;\n"
"    mov.u32 %r5, %ctaid.y;\n"
"    mov.u32 %r6, %ntid.y;\n"
"    mov.u32 %r7, %tid.y;\n"
"    mad.lo.u32 %r8, %r5, %r6, %r7;\n"
"    cvt.s64.s32 %rd8, %r8;\n"
"    setp.ge.s64 %p1, %rd8, %rd4;\n"
"    setp.ge.s64 %p2, %rd7, %rd6;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra DONE;\n"
"    mov.f32 %f1, 0f00000000;\n"
"    mov.s64 %rd9, 0;\n"
"LOOP:\n"
"    setp.ge.s64 %p4, %rd9, %rd5;\n"
"    @%p4 bra LOOP_END;\n"
"    mul.lo.s64 %rd10, %rd8, %rd5;\n"
"    add.s64 %rd11, %rd10, %rd9;\n"
"    shl.b64 %rd12, %rd11, 2;\n"
"    add.s64 %rd13, %rd1, %rd12;\n"
"    ld.global.f32 %f2, [%rd13];\n"
"    mul.lo.s64 %rd14, %rd9, %rd6;\n"
"    add.s64 %rd15, %rd14, %rd7;\n"
"    shl.b64 %rd16, %rd15, 2;\n"
"    add.s64 %rd17, %rd2, %rd16;\n"
"    ld.global.f32 %f3, [%rd17];\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"    add.s64 %rd9, %rd9, 1;\n"
"    bra LOOP;\n"
"LOOP_END:\n"
"    mul.lo.s64 %rd18, %rd8, %rd6;\n"
"    add.s64 %rd19, %rd18, %rd7;\n"
"    shl.b64 %rd20, %rd19, 2;\n"
"    add.s64 %rd21, %rd3, %rd20;\n"
"    st.global.f32 [%rd21], %f1;\n"
"DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_relu_kernel(\n"
"    .param .u64 param_in,\n"
"    .param .u64 param_out,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<10>;\n"
"    .reg .f32 %f<4>;\n"
"    .reg .pred %p<4>;\n"
"    ld.param.u64 %rd1, [param_in];\n"
"    ld.param.u64 %rd2, [param_out];\n"
"    ld.param.s64 %rd3, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd4, %r4;\n"
"    setp.ge.s64 %p1, %rd4, %rd3;\n"
"    @%p1 bra RELU_DONE;\n"
"    shl.b64 %rd5, %rd4, 2;\n"
"    add.s64 %rd6, %rd1, %rd5;\n"
"    ld.global.f32 %f1, [%rd6];\n"
"    mov.f32 %f2, 0f00000000;\n"
"    max.f32 %f3, %f1, %f2;\n"
"    add.s64 %rd7, %rd2, %rd5;\n"
"    st.global.f32 [%rd7], %f3;\n"
"RELU_DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_add_kernel(\n"
"    .param .u64 param_a,\n"
"    .param .u64 param_b,\n"
"    .param .u64 param_out,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<12>;\n"
"    .reg .f32 %f<4>;\n"
"    .reg .pred %p<4>;\n"
"    ld.param.u64 %rd1, [param_a];\n"
"    ld.param.u64 %rd2, [param_b];\n"
"    ld.param.u64 %rd3, [param_out];\n"
"    ld.param.s64 %rd4, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd5, %r4;\n"
"    setp.ge.s64 %p1, %rd5, %rd4;\n"
"    @%p1 bra ADD_DONE;\n"
"    shl.b64 %rd6, %rd5, 2;\n"
"    add.s64 %rd7, %rd1, %rd6;\n"
"    ld.global.f32 %f1, [%rd7];\n"
"    add.s64 %rd8, %rd2, %rd6;\n"
"    ld.global.f32 %f2, [%rd8];\n"
"    add.f32 %f3, %f1, %f2;\n"
"    add.s64 %rd9, %rd3, %rd6;\n"
"    st.global.f32 [%rd9], %f3;\n"
"ADD_DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_sub_kernel(\n"
"    .param .u64 param_a,\n"
"    .param .u64 param_b,\n"
"    .param .u64 param_out,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<12>;\n"
"    .reg .f32 %f<4>;\n"
"    .reg .pred %p<4>;\n"
"    ld.param.u64 %rd1, [param_a];\n"
"    ld.param.u64 %rd2, [param_b];\n"
"    ld.param.u64 %rd3, [param_out];\n"
"    ld.param.s64 %rd4, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd5, %r4;\n"
"    setp.ge.s64 %p1, %rd5, %rd4;\n"
"    @%p1 bra SUB_DONE;\n"
"    shl.b64 %rd6, %rd5, 2;\n"
"    add.s64 %rd7, %rd1, %rd6;\n"
"    ld.global.f32 %f1, [%rd7];\n"
"    add.s64 %rd8, %rd2, %rd6;\n"
"    ld.global.f32 %f2, [%rd8];\n"
"    sub.f32 %f3, %f1, %f2;\n"
"    add.s64 %rd9, %rd3, %rd6;\n"
"    st.global.f32 [%rd9], %f3;\n"
"SUB_DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_mul_kernel(\n"
"    .param .u64 param_a,\n"
"    .param .u64 param_b,\n"
"    .param .u64 param_out,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<12>;\n"
"    .reg .f32 %f<4>;\n"
"    .reg .pred %p<4>;\n"
"    ld.param.u64 %rd1, [param_a];\n"
"    ld.param.u64 %rd2, [param_b];\n"
"    ld.param.u64 %rd3, [param_out];\n"
"    ld.param.s64 %rd4, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd5, %r4;\n"
"    setp.ge.s64 %p1, %rd5, %rd4;\n"
"    @%p1 bra MUL_DONE;\n"
"    shl.b64 %rd6, %rd5, 2;\n"
"    add.s64 %rd7, %rd1, %rd6;\n"
"    ld.global.f32 %f1, [%rd7];\n"
"    add.s64 %rd8, %rd2, %rd6;\n"
"    ld.global.f32 %f2, [%rd8];\n"
"    mul.f32 %f3, %f1, %f2;\n"
"    add.s64 %rd9, %rd3, %rd6;\n"
"    st.global.f32 [%rd9], %f3;\n"
"MUL_DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_matmul_relu_kernel(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .s64 param_M,\n"
"    .param .s64 param_K,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<30>;\n"
"    .reg .f32 %f<10>;\n"
"    .reg .pred %p<10>;\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_C];\n"
"    ld.param.s64 %rd4, [param_M];\n"
"    ld.param.s64 %rd5, [param_K];\n"
"    ld.param.s64 %rd6, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd7, %r4;\n"
"    mov.u32 %r5, %ctaid.y;\n"
"    mov.u32 %r6, %ntid.y;\n"
"    mov.u32 %r7, %tid.y;\n"
"    mad.lo.u32 %r8, %r5, %r6, %r7;\n"
"    cvt.s64.s32 %rd8, %r8;\n"
"    setp.ge.s64 %p1, %rd8, %rd4;\n"
"    setp.ge.s64 %p2, %rd7, %rd6;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra FUSED_RELU_DONE;\n"
"    mov.f32 %f1, 0f00000000;\n"
"    mov.s64 %rd9, 0;\n"
"FUSED_RELU_LOOP:\n"
"    setp.ge.s64 %p4, %rd9, %rd5;\n"
"    @%p4 bra FUSED_RELU_LOOP_END;\n"
"    mul.lo.s64 %rd10, %rd8, %rd5;\n"
"    add.s64 %rd11, %rd10, %rd9;\n"
"    shl.b64 %rd12, %rd11, 2;\n"
"    add.s64 %rd13, %rd1, %rd12;\n"
"    ld.global.f32 %f2, [%rd13];\n"
"    mul.lo.s64 %rd14, %rd9, %rd6;\n"
"    add.s64 %rd15, %rd14, %rd7;\n"
"    shl.b64 %rd16, %rd15, 2;\n"
"    add.s64 %rd17, %rd2, %rd16;\n"
"    ld.global.f32 %f3, [%rd17];\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"    add.s64 %rd9, %rd9, 1;\n"
"    bra FUSED_RELU_LOOP;\n"
"FUSED_RELU_LOOP_END:\n"
"    mov.f32 %f4, 0f00000000;\n"
"    max.f32 %f5, %f1, %f4;\n"
"    mul.lo.s64 %rd18, %rd8, %rd6;\n"
"    add.s64 %rd19, %rd18, %rd7;\n"
"    shl.b64 %rd20, %rd19, 2;\n"
"    add.s64 %rd21, %rd3, %rd20;\n"
"    st.global.f32 [%rd21], %f5;\n"
"FUSED_RELU_DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_matmul_add_kernel(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_bias,\n"
"    .param .u64 param_C,\n"
"    .param .s64 param_M,\n"
"    .param .s64 param_K,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<32>;\n"
"    .reg .f32 %f<10>;\n"
"    .reg .pred %p<10>;\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_bias];\n"
"    ld.param.u64 %rd4, [param_C];\n"
"    ld.param.s64 %rd5, [param_M];\n"
"    ld.param.s64 %rd6, [param_K];\n"
"    ld.param.s64 %rd7, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd8, %r4;\n"
"    mov.u32 %r5, %ctaid.y;\n"
"    mov.u32 %r6, %ntid.y;\n"
"    mov.u32 %r7, %tid.y;\n"
"    mad.lo.u32 %r8, %r5, %r6, %r7;\n"
"    cvt.s64.s32 %rd9, %r8;\n"
"    setp.ge.s64 %p1, %rd9, %rd5;\n"
"    setp.ge.s64 %p2, %rd8, %rd7;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra FUSED_ADD_DONE;\n"
"    mov.f32 %f1, 0f00000000;\n"
"    mov.s64 %rd10, 0;\n"
"FUSED_ADD_LOOP:\n"
"    setp.ge.s64 %p4, %rd10, %rd6;\n"
"    @%p4 bra FUSED_ADD_LOOP_END;\n"
"    mul.lo.s64 %rd11, %rd9, %rd6;\n"
"    add.s64 %rd12, %rd11, %rd10;\n"
"    shl.b64 %rd13, %rd12, 2;\n"
"    add.s64 %rd14, %rd1, %rd13;\n"
"    ld.global.f32 %f2, [%rd14];\n"
"    mul.lo.s64 %rd15, %rd10, %rd7;\n"
"    add.s64 %rd16, %rd15, %rd8;\n"
"    shl.b64 %rd17, %rd16, 2;\n"
"    add.s64 %rd18, %rd2, %rd17;\n"
"    ld.global.f32 %f3, [%rd18];\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"    add.s64 %rd10, %rd10, 1;\n"
"    bra FUSED_ADD_LOOP;\n"
"FUSED_ADD_LOOP_END:\n"
"    shl.b64 %rd19, %rd8, 2;\n"
"    add.s64 %rd20, %rd3, %rd19;\n"
"    ld.global.f32 %f4, [%rd20];\n"
"    add.f32 %f5, %f1, %f4;\n"
"    mul.lo.s64 %rd21, %rd9, %rd7;\n"
"    add.s64 %rd22, %rd21, %rd8;\n"
"    shl.b64 %rd23, %rd22, 2;\n"
"    add.s64 %rd24, %rd4, %rd23;\n"
"    st.global.f32 [%rd24], %f5;\n"
"FUSED_ADD_DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_matmul_add_relu_kernel(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_bias,\n"
"    .param .u64 param_C,\n"
"    .param .s64 param_M,\n"
"    .param .s64 param_K,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<32>;\n"
"    .reg .f32 %f<10>;\n"
"    .reg .pred %p<10>;\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_bias];\n"
"    ld.param.u64 %rd4, [param_C];\n"
"    ld.param.s64 %rd5, [param_M];\n"
"    ld.param.s64 %rd6, [param_K];\n"
"    ld.param.s64 %rd7, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd8, %r4;\n"
"    mov.u32 %r5, %ctaid.y;\n"
"    mov.u32 %r6, %ntid.y;\n"
"    mov.u32 %r7, %tid.y;\n"
"    mad.lo.u32 %r8, %r5, %r6, %r7;\n"
"    cvt.s64.s32 %rd9, %r8;\n"
"    setp.ge.s64 %p1, %rd9, %rd5;\n"
"    setp.ge.s64 %p2, %rd8, %rd7;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra FUSED_ADD_RELU_DONE;\n"
"    mov.f32 %f1, 0f00000000;\n"
"    mov.s64 %rd10, 0;\n"
"FUSED_ADD_RELU_LOOP:\n"
"    setp.ge.s64 %p4, %rd10, %rd6;\n"
"    @%p4 bra FUSED_ADD_RELU_LOOP_END;\n"
"    mul.lo.s64 %rd11, %rd9, %rd6;\n"
"    add.s64 %rd12, %rd11, %rd10;\n"
"    shl.b64 %rd13, %rd12, 2;\n"
"    add.s64 %rd14, %rd1, %rd13;\n"
"    ld.global.f32 %f2, [%rd14];\n"
"    mul.lo.s64 %rd15, %rd10, %rd7;\n"
"    add.s64 %rd16, %rd15, %rd8;\n"
"    shl.b64 %rd17, %rd16, 2;\n"
"    add.s64 %rd18, %rd2, %rd17;\n"
"    ld.global.f32 %f3, [%rd18];\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"    add.s64 %rd10, %rd10, 1;\n"
"    bra FUSED_ADD_RELU_LOOP;\n"
"FUSED_ADD_RELU_LOOP_END:\n"
"    shl.b64 %rd19, %rd8, 2;\n"
"    add.s64 %rd20, %rd3, %rd19;\n"
"    ld.global.f32 %f4, [%rd20];\n"
"    add.f32 %f5, %f1, %f4;\n"
"    mov.f32 %f6, 0f00000000;\n"
"    max.f32 %f7, %f5, %f6;\n"
"    mul.lo.s64 %rd21, %rd9, %rd7;\n"
"    add.s64 %rd22, %rd21, %rd8;\n"
"    shl.b64 %rd23, %rd22, 2;\n"
"    add.s64 %rd24, %rd4, %rd23;\n"
"    st.global.f32 [%rd24], %f7;\n"
"FUSED_ADD_RELU_DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_matmul_add_relu_residual_kernel(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_bias,\n"
"    .param .u64 param_residual,\n"
"    .param .u64 param_C,\n"
"    .param .s64 param_M,\n"
"    .param .s64 param_K,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<36>;\n"
"    .reg .f32 %f<12>;\n"
"    .reg .pred %p<10>;\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_bias];\n"
"    ld.param.u64 %rd4, [param_residual];\n"
"    ld.param.u64 %rd5, [param_C];\n"
"    ld.param.s64 %rd6, [param_M];\n"
"    ld.param.s64 %rd7, [param_K];\n"
"    ld.param.s64 %rd8, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd9, %r4;\n"
"    mov.u32 %r5, %ctaid.y;\n"
"    mov.u32 %r6, %ntid.y;\n"
"    mov.u32 %r7, %tid.y;\n"
"    mad.lo.u32 %r8, %r5, %r6, %r7;\n"
"    cvt.s64.s32 %rd10, %r8;\n"
"    setp.ge.s64 %p1, %rd10, %rd6;\n"
"    setp.ge.s64 %p2, %rd9, %rd8;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra FUSED_RESIDUAL_DONE;\n"
"    mov.f32 %f1, 0f00000000;\n"
"    mov.s64 %rd11, 0;\n"
"FUSED_RESIDUAL_LOOP:\n"
"    setp.ge.s64 %p4, %rd11, %rd7;\n"
"    @%p4 bra FUSED_RESIDUAL_LOOP_END;\n"
"    mul.lo.s64 %rd12, %rd10, %rd7;\n"
"    add.s64 %rd13, %rd12, %rd11;\n"
"    shl.b64 %rd14, %rd13, 2;\n"
"    add.s64 %rd15, %rd1, %rd14;\n"
"    ld.global.f32 %f2, [%rd15];\n"
"    mul.lo.s64 %rd16, %rd11, %rd8;\n"
"    add.s64 %rd17, %rd16, %rd9;\n"
"    shl.b64 %rd18, %rd17, 2;\n"
"    add.s64 %rd19, %rd2, %rd18;\n"
"    ld.global.f32 %f3, [%rd19];\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"    add.s64 %rd11, %rd11, 1;\n"
"    bra FUSED_RESIDUAL_LOOP;\n"
"FUSED_RESIDUAL_LOOP_END:\n"
"    shl.b64 %rd20, %rd9, 2;\n"
"    add.s64 %rd21, %rd3, %rd20;\n"
"    ld.global.f32 %f4, [%rd21];\n"
"    add.f32 %f5, %f1, %f4;\n"
"    mov.f32 %f6, 0f00000000;\n"
"    max.f32 %f7, %f5, %f6;\n"
"    mul.lo.s64 %rd22, %rd10, %rd8;\n"
"    add.s64 %rd23, %rd22, %rd9;\n"
"    shl.b64 %rd24, %rd23, 2;\n"
"    add.s64 %rd25, %rd4, %rd24;\n"
"    ld.global.f32 %f8, [%rd25];\n"
"    add.f32 %f9, %f7, %f8;\n"
"    add.s64 %rd26, %rd5, %rd24;\n"
"    st.global.f32 [%rd26], %f9;\n"
"FUSED_RESIDUAL_DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry gpu_add_relu_kernel(\n"
"    .param .u64 param_a,\n"
"    .param .u64 param_b,\n"
"    .param .u64 param_out,\n"
"    .param .s64 param_N\n"
") {\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<12>;\n"
"    .reg .f32 %f<6>;\n"
"    .reg .pred %p<4>;\n"
"    ld.param.u64 %rd1, [param_a];\n"
"    ld.param.u64 %rd2, [param_b];\n"
"    ld.param.u64 %rd3, [param_out];\n"
"    ld.param.s64 %rd4, [param_N];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd5, %r4;\n"
"    setp.ge.s64 %p1, %rd5, %rd4;\n"
"    @%p1 bra ADD_RELU_DONE;\n"
"    shl.b64 %rd6, %rd5, 2;\n"
"    add.s64 %rd7, %rd1, %rd6;\n"
"    ld.global.f32 %f1, [%rd7];\n"
"    add.s64 %rd8, %rd2, %rd6;\n"
"    ld.global.f32 %f2, [%rd8];\n"
"    add.f32 %f3, %f1, %f2;\n"
"    mov.f32 %f4, 0f00000000;\n"
"    max.f32 %f5, %f3, %f4;\n"
"    add.s64 %rd9, %rd3, %rd6;\n"
"    st.global.f32 [%rd9], %f5;\n"
"ADD_RELU_DONE:\n"
"    ret;\n"
"}\n";

void init_driver() {
    if (g_cuda.initialized) return;
    g_cuda.initialized = true;

#if defined(_WIN32)
    HMODULE hLib = LoadLibraryA("nvcuda.dll");
    if (!hLib) return;

    g_cuda.cuInit              = (PFN_cuInit)GetProcAddress(hLib, "cuInit");
    g_cuda.cuDeviceGet         = (PFN_cuDeviceGet)GetProcAddress(hLib, "cuDeviceGet");
    g_cuda.cuCtxCreate         = (PFN_cuCtxCreate)GetProcAddress(hLib, "cuCtxCreate_v2");
    if (!g_cuda.cuCtxCreate)
        g_cuda.cuCtxCreate     = (PFN_cuCtxCreate)GetProcAddress(hLib, "cuCtxCreate");
    g_cuda.cuMemAlloc          = (PFN_cuMemAlloc)GetProcAddress(hLib, "cuMemAlloc_v2");
    if (!g_cuda.cuMemAlloc)
        g_cuda.cuMemAlloc      = (PFN_cuMemAlloc)GetProcAddress(hLib, "cuMemAlloc");
    g_cuda.cuMemFree           = (PFN_cuMemFree)GetProcAddress(hLib, "cuMemFree_v2");
    if (!g_cuda.cuMemFree)
        g_cuda.cuMemFree       = (PFN_cuMemFree)GetProcAddress(hLib, "cuMemFree");
    g_cuda.cuMemcpyHtoD        = (PFN_cuMemcpyHtoD)GetProcAddress(hLib, "cuMemcpyHtoD_v2");
    if (!g_cuda.cuMemcpyHtoD)
        g_cuda.cuMemcpyHtoD    = (PFN_cuMemcpyHtoD)GetProcAddress(hLib, "cuMemcpyHtoD");
    g_cuda.cuMemcpyDtoH        = (PFN_cuMemcpyDtoH)GetProcAddress(hLib, "cuMemcpyDtoH_v2");
    if (!g_cuda.cuMemcpyDtoH)
        g_cuda.cuMemcpyDtoH    = (PFN_cuMemcpyDtoH)GetProcAddress(hLib, "cuMemcpyDtoH");
    g_cuda.cuModuleLoadData    = (PFN_cuModuleLoadData)GetProcAddress(hLib, "cuModuleLoadData");
    g_cuda.cuModuleGetFunction = (PFN_cuModuleGetFunction)GetProcAddress(hLib, "cuModuleGetFunction");
    g_cuda.cuLaunchKernel      = (PFN_cuLaunchKernel)GetProcAddress(hLib, "cuLaunchKernel");
    g_cuda.cuCtxSynchronize    = (PFN_cuCtxSynchronize)GetProcAddress(hLib, "cuCtxSynchronize");
    g_cuda.cuDeviceGetName     = (PFN_cuDeviceGetName)GetProcAddress(hLib, "cuDeviceGetName");
    g_cuda.cuDeviceGetAttribute = (PFN_cuDeviceGetAttribute)GetProcAddress(hLib, "cuDeviceGetAttribute");
    g_cuda.cuMemGetInfo        = (PFN_cuMemGetInfo)GetProcAddress(hLib, "cuMemGetInfo");
#else
    void* hLib = dlopen("libcuda.so", RTLD_LAZY);
    if (!hLib) hLib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!hLib) return;

    g_cuda.cuInit              = (PFN_cuInit)dlsym(hLib, "cuInit");
    g_cuda.cuDeviceGet         = (PFN_cuDeviceGet)dlsym(hLib, "cuDeviceGet");
    g_cuda.cuCtxCreate         = (PFN_cuCtxCreate)dlsym(hLib, "cuCtxCreate_v2");
    if (!g_cuda.cuCtxCreate)
        g_cuda.cuCtxCreate     = (PFN_cuCtxCreate)dlsym(hLib, "cuCtxCreate");
    g_cuda.cuMemAlloc          = (PFN_cuMemAlloc)dlsym(hLib, "cuMemAlloc_v2");
    if (!g_cuda.cuMemAlloc)
        g_cuda.cuMemAlloc      = (PFN_cuMemAlloc)dlsym(hLib, "cuMemAlloc");
    g_cuda.cuMemFree           = (PFN_cuMemFree)dlsym(hLib, "cuMemFree_v2");
    if (!g_cuda.cuMemFree)
        g_cuda.cuMemFree       = (PFN_cuMemFree)dlsym(hLib, "cuMemFree");
    g_cuda.cuMemcpyHtoD        = (PFN_cuMemcpyHtoD)dlsym(hLib, "cuMemcpyHtoD_v2");
    if (!g_cuda.cuMemcpyHtoD)
        g_cuda.cuMemcpyHtoD    = (PFN_cuMemcpyHtoD)dlsym(hLib, "cuMemcpyHtoD");
    g_cuda.cuMemcpyDtoH        = (PFN_cuMemcpyDtoH)dlsym(hLib, "cuMemcpyDtoH_v2");
    if (!g_cuda.cuMemcpyDtoH)
        g_cuda.cuMemcpyDtoH    = (PFN_cuMemcpyDtoH)dlsym(hLib, "cuMemcpyDtoH");
    g_cuda.cuModuleLoadData    = (PFN_cuModuleLoadData)dlsym(hLib, "cuModuleLoadData");
    g_cuda.cuModuleGetFunction = (PFN_cuModuleGetFunction)dlsym(hLib, "cuModuleGetFunction");
    g_cuda.cuLaunchKernel      = (PFN_cuLaunchKernel)dlsym(hLib, "cuLaunchKernel");
    g_cuda.cuCtxSynchronize    = (PFN_cuCtxSynchronize)dlsym(hLib, "cuCtxSynchronize");
    g_cuda.cuDeviceGetName     = (PFN_cuDeviceGetName)dlsym(hLib, "cuDeviceGetName");
    g_cuda.cuDeviceGetAttribute = (PFN_cuDeviceGetAttribute)dlsym(hLib, "cuDeviceGetAttribute");
    g_cuda.cuMemGetInfo        = (PFN_cuMemGetInfo)dlsym(hLib, "cuMemGetInfo");
#endif

    if (!g_cuda.cuInit || !g_cuda.cuDeviceGet || !g_cuda.cuCtxCreate || !g_cuda.cuMemAlloc ||
        !g_cuda.cuMemFree || !g_cuda.cuMemcpyHtoD || !g_cuda.cuMemcpyDtoH ||
        !g_cuda.cuModuleLoadData || !g_cuda.cuModuleGetFunction || !g_cuda.cuLaunchKernel) {
        return;
    }

    if (g_cuda.cuInit(0) != CUDA_SUCCESS) return;

    CUdevice dev = 0;
    if (g_cuda.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return;
    if (g_cuda.cuCtxCreate(&g_cuda.context, 0, dev) != CUDA_SUCCESS) return;

    if (g_cuda.cuModuleLoadData(&g_cuda.module, kPtxSource) != CUDA_SUCCESS) return;

    g_cuda.cuModuleGetFunction(&g_cuda.fn_matmul, g_cuda.module, "gpu_matmul_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_relu, g_cuda.module, "gpu_relu_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_add, g_cuda.module, "gpu_add_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_sub, g_cuda.module, "gpu_sub_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_mul, g_cuda.module, "gpu_mul_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_matmul_relu, g_cuda.module, "gpu_matmul_relu_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_matmul_add, g_cuda.module, "gpu_matmul_add_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_matmul_add_relu, g_cuda.module, "gpu_matmul_add_relu_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_matmul_add_relu_residual, g_cuda.module, "gpu_matmul_add_relu_residual_kernel");
    g_cuda.cuModuleGetFunction(&g_cuda.fn_add_relu, g_cuda.module, "gpu_add_relu_kernel");

    g_cuda.available = (g_cuda.fn_matmul && g_cuda.fn_relu && g_cuda.fn_add &&
                        g_cuda.fn_matmul_relu && g_cuda.fn_matmul_add && g_cuda.fn_matmul_add_relu &&
                        g_cuda.fn_matmul_add_relu_residual && g_cuda.fn_add_relu);
}

}

extern "C" {

bool forge_gpu_is_available() {
    init_driver();
    return g_cuda.available;
}

void forge_gpu_init() {
    init_driver();
}

void* forge_gpu_malloc(size_t bytes) {
    init_driver();
    if (!g_cuda.available) {
        return std::malloc(bytes);
    }
    CUdeviceptr dptr = 0;
    if (g_cuda.cuMemAlloc(&dptr, bytes) == CUDA_SUCCESS) {
        return reinterpret_cast<void*>(dptr);
    }
    return nullptr;
}

void forge_gpu_free(void* ptr) {
    if (!ptr) return;
    init_driver();
    if (!g_cuda.available) {
        std::free(ptr);
        return;
    }
    g_cuda.cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
}

void forge_gpu_memcpy_to_device(void* dst, const void* src, size_t bytes) {
    if (!dst || !src || bytes == 0) return;
    init_driver();
    if (!g_cuda.available) {
        std::memcpy(dst, src, bytes);
        return;
    }
    g_cuda.cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, bytes);
}

void forge_gpu_memcpy_to_host(void* dst, const void* src, size_t bytes) {
    if (!dst || !src || bytes == 0) return;
    init_driver();
    if (!g_cuda.available) {
        std::memcpy(dst, src, bytes);
        return;
    }
    g_cuda.cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), bytes);
}

void forge_gpu_sync() {
    init_driver();
    if (g_cuda.available && g_cuda.cuCtxSynchronize) {
        g_cuda.cuCtxSynchronize();
    }
}

void forge_gpu_matmul(const float* A, const float* B, float* C,
                      int64_t M, int64_t K, int64_t N) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_A = reinterpret_cast<CUdeviceptr>(A);
    CUdeviceptr d_B = reinterpret_cast<CUdeviceptr>(B);
    CUdeviceptr d_C = reinterpret_cast<CUdeviceptr>(C);

    unsigned int blockX = 16;
    unsigned int blockY = 16;
    unsigned int gridX  = (static_cast<unsigned int>(N) + blockX - 1) / blockX;
    unsigned int gridY  = (static_cast<unsigned int>(M) + blockY - 1) / blockY;

    void* args[] = { &d_A, &d_B, &d_C, &M, &K, &N };
    g_cuda.cuLaunchKernel(g_cuda.fn_matmul, gridX, gridY, 1, blockX, blockY, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_relu(const float* in, float* out, int64_t n) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_in  = reinterpret_cast<CUdeviceptr>(in);
    CUdeviceptr d_out = reinterpret_cast<CUdeviceptr>(out);

    unsigned int blockSize = 256;
    unsigned int gridSize  = (static_cast<unsigned int>(n) + blockSize - 1) / blockSize;

    void* args[] = { &d_in, &d_out, &n };
    g_cuda.cuLaunchKernel(g_cuda.fn_relu, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_add(const float* a, const float* b, float* out, int64_t n) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_a   = reinterpret_cast<CUdeviceptr>(a);
    CUdeviceptr d_b   = reinterpret_cast<CUdeviceptr>(b);
    CUdeviceptr d_out = reinterpret_cast<CUdeviceptr>(out);

    unsigned int blockSize = 256;
    unsigned int gridSize  = (static_cast<unsigned int>(n) + blockSize - 1) / blockSize;

    void* args[] = { &d_a, &d_b, &d_out, &n };
    g_cuda.cuLaunchKernel(g_cuda.fn_add, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_sub(const float* a, const float* b, float* out, int64_t n) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_a   = reinterpret_cast<CUdeviceptr>(a);
    CUdeviceptr d_b   = reinterpret_cast<CUdeviceptr>(b);
    CUdeviceptr d_out = reinterpret_cast<CUdeviceptr>(out);

    unsigned int blockSize = 256;
    unsigned int gridSize  = (static_cast<unsigned int>(n) + blockSize - 1) / blockSize;

    void* args[] = { &d_a, &d_b, &d_out, &n };
    g_cuda.cuLaunchKernel(g_cuda.fn_sub, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_mul(const float* a, const float* b, float* out, int64_t n) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_a   = reinterpret_cast<CUdeviceptr>(a);
    CUdeviceptr d_b   = reinterpret_cast<CUdeviceptr>(b);
    CUdeviceptr d_out = reinterpret_cast<CUdeviceptr>(out);

    unsigned int blockSize = 256;
    unsigned int gridSize  = (static_cast<unsigned int>(n) + blockSize - 1) / blockSize;

    void* args[] = { &d_a, &d_b, &d_out, &n };
    g_cuda.cuLaunchKernel(g_cuda.fn_mul, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_matmul_relu_fused(const float* A, const float* B, float* C,
                                 int64_t M, int64_t K, int64_t N) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_A = reinterpret_cast<CUdeviceptr>(A);
    CUdeviceptr d_B = reinterpret_cast<CUdeviceptr>(B);
    CUdeviceptr d_C = reinterpret_cast<CUdeviceptr>(C);

    unsigned int blockX = 16;
    unsigned int blockY = 16;
    unsigned int gridX  = (static_cast<unsigned int>(N) + blockX - 1) / blockX;
    unsigned int gridY  = (static_cast<unsigned int>(M) + blockY - 1) / blockY;

    void* args[] = { &d_A, &d_B, &d_C, &M, &K, &N };
    g_cuda.cuLaunchKernel(g_cuda.fn_matmul_relu, gridX, gridY, 1, blockX, blockY, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_matmul_add_fused(const float* A, const float* B, const float* bias, float* C,
                                int64_t M, int64_t K, int64_t N) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_A    = reinterpret_cast<CUdeviceptr>(A);
    CUdeviceptr d_B    = reinterpret_cast<CUdeviceptr>(B);
    CUdeviceptr d_bias = reinterpret_cast<CUdeviceptr>(bias);
    CUdeviceptr d_C    = reinterpret_cast<CUdeviceptr>(C);

    unsigned int blockX = 16;
    unsigned int blockY = 16;
    unsigned int gridX  = (static_cast<unsigned int>(N) + blockX - 1) / blockX;
    unsigned int gridY  = (static_cast<unsigned int>(M) + blockY - 1) / blockY;

    void* args[] = { &d_A, &d_B, &d_bias, &d_C, &M, &K, &N };
    g_cuda.cuLaunchKernel(g_cuda.fn_matmul_add, gridX, gridY, 1, blockX, blockY, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_matmul_add_relu_fused(const float* A, const float* B, const float* bias, float* C,
                                     int64_t M, int64_t K, int64_t N) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_A    = reinterpret_cast<CUdeviceptr>(A);
    CUdeviceptr d_B    = reinterpret_cast<CUdeviceptr>(B);
    CUdeviceptr d_bias = reinterpret_cast<CUdeviceptr>(bias);
    CUdeviceptr d_C    = reinterpret_cast<CUdeviceptr>(C);

    unsigned int blockX = 16;
    unsigned int blockY = 16;
    unsigned int gridX  = (static_cast<unsigned int>(N) + blockX - 1) / blockX;
    unsigned int gridY  = (static_cast<unsigned int>(M) + blockY - 1) / blockY;

    void* args[] = { &d_A, &d_B, &d_bias, &d_C, &M, &K, &N };
    g_cuda.cuLaunchKernel(g_cuda.fn_matmul_add_relu, gridX, gridY, 1, blockX, blockY, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_matmul_add_relu_residual_fused(const float* A, const float* B, const float* bias, const float* residual, float* C,
                                              int64_t M, int64_t K, int64_t N) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_A        = reinterpret_cast<CUdeviceptr>(A);
    CUdeviceptr d_B        = reinterpret_cast<CUdeviceptr>(B);
    CUdeviceptr d_bias     = reinterpret_cast<CUdeviceptr>(bias);
    CUdeviceptr d_residual = reinterpret_cast<CUdeviceptr>(residual);
    CUdeviceptr d_C        = reinterpret_cast<CUdeviceptr>(C);

    unsigned int blockX = 16;
    unsigned int blockY = 16;
    unsigned int gridX  = (static_cast<unsigned int>(N) + blockX - 1) / blockX;
    unsigned int gridY  = (static_cast<unsigned int>(M) + blockY - 1) / blockY;

    void* args[] = { &d_A, &d_B, &d_bias, &d_residual, &d_C, &M, &K, &N };
    g_cuda.cuLaunchKernel(g_cuda.fn_matmul_add_relu_residual, gridX, gridY, 1, blockX, blockY, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_add_relu_fused(const float* a, const float* b, float* out, int64_t n) {
    init_driver();
    if (!g_cuda.available) return;

    CUdeviceptr d_a   = reinterpret_cast<CUdeviceptr>(a);
    CUdeviceptr d_b   = reinterpret_cast<CUdeviceptr>(b);
    CUdeviceptr d_out = reinterpret_cast<CUdeviceptr>(out);

    unsigned int blockSize = 256;
    unsigned int gridSize  = (static_cast<unsigned int>(n) + blockSize - 1) / blockSize;

    void* args[] = { &d_a, &d_b, &d_out, &n };
    g_cuda.cuLaunchKernel(g_cuda.fn_add_relu, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
}

bool forge_gpu_get_device_name(char* out_name, int max_len) {
    init_driver();
    if (!g_cuda.available || !g_cuda.cuDeviceGetName) return false;
    CUdevice dev = 0;
    if (g_cuda.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;
    return g_cuda.cuDeviceGetName(out_name, max_len, dev) == CUDA_SUCCESS;
}

int forge_gpu_get_attribute(int attribute) {
    init_driver();
    if (!g_cuda.available || !g_cuda.cuDeviceGetAttribute) return -1;
    CUdevice dev = 0;
    if (g_cuda.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return -1;
    int val = 0;
    if (g_cuda.cuDeviceGetAttribute(&val, attribute, dev) == CUDA_SUCCESS) {
        return val;
    }
    return -1;
}

bool forge_gpu_get_memory_info(size_t* free_bytes, size_t* total_bytes) {
    init_driver();
    if (!g_cuda.available || !g_cuda.cuMemGetInfo) return false;
    return g_cuda.cuMemGetInfo(free_bytes, total_bytes) == CUDA_SUCCESS;
}

void forge_gpu_auto_tune_elementwise(int64_t n, unsigned int* block_size, unsigned int* grid_size) {
    if (!block_size || !grid_size) return;
    *block_size = 256;
    *grid_size = static_cast<unsigned int>((n + 255) / 256);
    if (*grid_size == 0) *grid_size = 1;
}

void forge_gpu_auto_tune_matmul(int64_t M, int64_t N, unsigned int* block_x, unsigned int* block_y, unsigned int* grid_x, unsigned int* grid_y) {
    if (!block_x || !block_y || !grid_x || !grid_y) return;
    *block_x = 16;
    *block_y = 16;
    *grid_x = static_cast<unsigned int>((N + 15) / 16);
    *grid_y = static_cast<unsigned int>((M + 15) / 16);
    if (*grid_x == 0) *grid_x = 1;
    if (*grid_y == 0) *grid_y = 1;
}

void forge_gpu_relu_backward(const float* grad_out, const float* in, float* grad_in, int64_t n) {
    init_driver();
    if (!g_cuda.available) return;
    std::vector<float> h_go(n);
    std::vector<float> h_in(n);
    std::vector<float> h_gi(n);
    forge_gpu_memcpy_to_host(h_go.data(), grad_out, n * sizeof(float));
    forge_gpu_memcpy_to_host(h_in.data(), in, n * sizeof(float));
    for (int64_t i = 0; i < n; ++i) {
        h_gi[i] = (h_in[i] > 0.0f) ? h_go[i] : 0.0f;
    }
    forge_gpu_memcpy_to_device(grad_in, h_gi.data(), n * sizeof(float));
}

void forge_gpu_sgd_step(float* weights, const float* grads, float* velocity,
                        float lr, float momentum, float weight_decay, int64_t n) {
    init_driver();
    if (!g_cuda.available) return;
    std::vector<float> h_w(n);
    std::vector<float> h_g(n);
    std::vector<float> h_v(n, 0.0f);
    forge_gpu_memcpy_to_host(h_w.data(), weights, n * sizeof(float));
    forge_gpu_memcpy_to_host(h_g.data(), grads, n * sizeof(float));
    if (velocity && momentum > 0.0f) {
        forge_gpu_memcpy_to_host(h_v.data(), velocity, n * sizeof(float));
    }
    for (int64_t i = 0; i < n; ++i) {
        float g = h_g[i];
        if (weight_decay != 0.0f) {
            g += weight_decay * h_w[i];
        }
        if (velocity && momentum > 0.0f) {
            h_v[i] = momentum * h_v[i] + g;
            h_w[i] -= lr * h_v[i];
        } else {
            h_w[i] -= lr * g;
        }
    }
    forge_gpu_memcpy_to_device(weights, h_w.data(), n * sizeof(float));
    if (velocity && momentum > 0.0f) {
        forge_gpu_memcpy_to_device(velocity, h_v.data(), n * sizeof(float));
    }
}

void forge_gpu_adam_step(float* weights, const float* grads, float* m, float* v,
                         float lr, float beta1, float beta2, float eps,
                         float weight_decay, int64_t step, int64_t n) {
    init_driver();
    if (!g_cuda.available) return;
    std::vector<float> h_w(n);
    std::vector<float> h_g(n);
    std::vector<float> h_m(n);
    std::vector<float> h_v(n);
    forge_gpu_memcpy_to_host(h_w.data(), weights, n * sizeof(float));
    forge_gpu_memcpy_to_host(h_g.data(), grads, n * sizeof(float));
    forge_gpu_memcpy_to_host(h_m.data(), m, n * sizeof(float));
    forge_gpu_memcpy_to_host(h_v.data(), v, n * sizeof(float));
    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    for (int64_t i = 0; i < n; ++i) {
        float g = h_g[i];
        if (weight_decay != 0.0f) {
            g += weight_decay * h_w[i];
        }
        h_m[i] = beta1 * h_m[i] + (1.0f - beta1) * g;
        h_v[i] = beta2 * h_v[i] + (1.0f - beta2) * (g * g);
        float m_hat = h_m[i] / bias_correction1;
        float v_hat = h_v[i] / bias_correction2;
        h_w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
    forge_gpu_memcpy_to_device(weights, h_w.data(), n * sizeof(float));
    forge_gpu_memcpy_to_device(m, h_m.data(), n * sizeof(float));
    forge_gpu_memcpy_to_device(v, h_v.data(), n * sizeof(float));
}

}
