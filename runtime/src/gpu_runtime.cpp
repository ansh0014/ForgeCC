#include "forgecc_rt/gpu_runtime.hpp"
#include "forgecc_rt/tensor.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>

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

    bool initialized = false;
    bool available = false;
    CUcontext context = nullptr;
    CUmodule  module = nullptr;
    CUfunction fn_matmul = nullptr;
    CUfunction fn_relu = nullptr;
    CUfunction fn_matmul_relu = nullptr;
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
"\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_C];\n"
"    ld.param.s64 %rd4, [param_M];\n"
"    ld.param.s64 %rd5, [param_K];\n"
"    ld.param.s64 %rd6, [param_N];\n"
"\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd7, %r4;\n"
"\n"
"    mov.u32 %r5, %ctaid.y;\n"
"    mov.u32 %r6, %ntid.y;\n"
"    mov.u32 %r7, %tid.y;\n"
"    mad.lo.u32 %r8, %r5, %r6, %r7;\n"
"    cvt.s64.s32 %rd8, %r8;\n"
"\n"
"    setp.ge.s64 %p1, %rd8, %rd4;\n"
"    setp.ge.s64 %p2, %rd7, %rd6;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra DONE;\n"
"\n"
"    mov.f32 %f1, 0f00000000;\n"
"    mov.s64 %rd9, 0;\n"
"\n"
"LOOP:\n"
"    setp.ge.s64 %p4, %rd9, %rd5;\n"
"    @%p4 bra LOOP_END;\n"
"\n"
"    mul.lo.s64 %rd10, %rd8, %rd5;\n"
"    add.s64 %rd11, %rd10, %rd9;\n"
"    shl.b64 %rd12, %rd11, 2;\n"
"    add.s64 %rd13, %rd1, %rd12;\n"
"    ld.global.f32 %f2, [%rd13];\n"
"\n"
"    mul.lo.s64 %rd14, %rd9, %rd6;\n"
"    add.s64 %rd15, %rd14, %rd7;\n"
"    shl.b64 %rd16, %rd15, 2;\n"
"    add.s64 %rd17, %rd2, %rd16;\n"
"    ld.global.f32 %f3, [%rd17];\n"
"\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"    add.s64 %rd9, %rd9, 1;\n"
"    bra LOOP;\n"
"\n"
"LOOP_END:\n"
"    mul.lo.s64 %rd18, %rd8, %rd6;\n"
"    add.s64 %rd19, %rd18, %rd7;\n"
"    shl.b64 %rd20, %rd19, 2;\n"
"    add.s64 %rd21, %rd3, %rd20;\n"
"    st.global.f32 [%rd21], %f1;\n"
"\n"
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
"\n"
"    ld.param.u64 %rd1, [param_in];\n"
"    ld.param.u64 %rd2, [param_out];\n"
"    ld.param.s64 %rd3, [param_N];\n"
"\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd4, %r4;\n"
"\n"
"    setp.ge.s64 %p1, %rd4, %rd3;\n"
"    @%p1 bra RELU_DONE;\n"
"\n"
"    shl.b64 %rd5, %rd4, 2;\n"
"    add.s64 %rd6, %rd1, %rd5;\n"
"    ld.global.f32 %f1, [%rd6];\n"
"    mov.f32 %f2, 0f00000000;\n"
"    max.f32 %f3, %f1, %f2;\n"
"    add.s64 %rd7, %rd2, %rd5;\n"
"    st.global.f32 [%rd7], %f3;\n"
"\n"
"RELU_DONE:\n"
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
"\n"
"    ld.param.u64 %rd1, [param_A];\n"
"    ld.param.u64 %rd2, [param_B];\n"
"    ld.param.u64 %rd3, [param_C];\n"
"    ld.param.s64 %rd4, [param_M];\n"
"    ld.param.s64 %rd5, [param_K];\n"
"    ld.param.s64 %rd6, [param_N];\n"
"\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
"    cvt.s64.s32 %rd7, %r4;\n"
"\n"
"    mov.u32 %r5, %ctaid.y;\n"
"    mov.u32 %r6, %ntid.y;\n"
"    mov.u32 %r7, %tid.y;\n"
"    mad.lo.u32 %r8, %r5, %r6, %r7;\n"
"    cvt.s64.s32 %rd8, %r8;\n"
"\n"
"    setp.ge.s64 %p1, %rd8, %rd4;\n"
"    setp.ge.s64 %p2, %rd7, %rd6;\n"
"    or.pred %p3, %p1, %p2;\n"
"    @%p3 bra FUSED_DONE;\n"
"\n"
"    mov.f32 %f1, 0f00000000;\n"
"    mov.s64 %rd9, 0;\n"
"\n"
"FUSED_LOOP:\n"
"    setp.ge.s64 %p4, %rd9, %rd5;\n"
"    @%p4 bra FUSED_LOOP_END;\n"
"\n"
"    mul.lo.s64 %rd10, %rd8, %rd5;\n"
"    add.s64 %rd11, %rd10, %rd9;\n"
"    shl.b64 %rd12, %rd11, 2;\n"
"    add.s64 %rd13, %rd1, %rd12;\n"
"    ld.global.f32 %f2, [%rd13];\n"
"\n"
"    mul.lo.s64 %rd14, %rd9, %rd6;\n"
"    add.s64 %rd15, %rd14, %rd7;\n"
"    shl.b64 %rd16, %rd15, 2;\n"
"    add.s64 %rd17, %rd2, %rd16;\n"
"    ld.global.f32 %f3, [%rd17];\n"
"\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"    add.s64 %rd9, %rd9, 1;\n"
"    bra FUSED_LOOP;\n"
"\n"
"FUSED_LOOP_END:\n"
"    mov.f32 %f4, 0f00000000;\n"
"    max.f32 %f5, %f1, %f4;\n"
"    mul.lo.s64 %rd18, %rd8, %rd6;\n"
"    add.s64 %rd19, %rd18, %rd7;\n"
"    shl.b64 %rd20, %rd19, 2;\n"
"    add.s64 %rd21, %rd3, %rd20;\n"
"    st.global.f32 [%rd21], %f5;\n"
"\n"
"FUSED_DONE:\n"
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
    g_cuda.cuModuleGetFunction(&g_cuda.fn_matmul_relu, g_cuda.module, "gpu_matmul_relu_kernel");

    g_cuda.available = (g_cuda.fn_matmul && g_cuda.fn_relu && g_cuda.fn_matmul_relu);
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

void forge_gpu_matmul(const float* A, const float* B, float* C,
                      int64_t M, int64_t K, int64_t N) {
    init_driver();
    if (!g_cuda.available) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        return;
    }

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
    if (!g_cuda.available) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
        }
        return;
    }

    CUdeviceptr d_in  = reinterpret_cast<CUdeviceptr>(in);
    CUdeviceptr d_out = reinterpret_cast<CUdeviceptr>(out);

    unsigned int blockSize = 256;
    unsigned int gridSize  = (static_cast<unsigned int>(n) + blockSize - 1) / blockSize;

    void* args[] = { &d_in, &d_out, &n };
    g_cuda.cuLaunchKernel(g_cuda.fn_relu, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
}

void forge_gpu_matmul_relu_fused(const float* A, const float* B, float* C,
                                 int64_t M, int64_t K, int64_t N) {
    init_driver();
    if (!g_cuda.available) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
        return;
    }

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

void forge_gpu_sync() {
    init_driver();
    if (g_cuda.available && g_cuda.cuCtxSynchronize) {
        g_cuda.cuCtxSynchronize();
    }
}

}

