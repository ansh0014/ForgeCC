#include "forgecc_rt/tensor.hpp"
#include <cstdint>

extern "C" {

void cpu_matmul(const float* A, const float* B, float* C,
                int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void cpu_relu(const float* in, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
    }
}

}
